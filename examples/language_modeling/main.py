from functools import partial
import fire
import datasets
import numpy as np

import jax
import jax.numpy as jnp
import optax

from transformers import \
    AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig

from redco import Deployer, Trainer, Predictor


def collate_fn(examples, tokenizer, text_key, max_length):
    batch = tokenizer(
        [(example[text_key] + tokenizer.eos_token) for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='np')

    batch['labels'] = np.copy(batch['input_ids'])
    batch['input_ids'][:, 1:] = batch['input_ids'][:, :-1]
    batch['input_ids'][:, 0] = tokenizer.eos_token_id

    return batch


def loss_fn(train_rng, state, params, batch, is_training, model_type):
    labels = batch.pop("labels")
    label_weights = batch['attention_mask']

    if model_type != 'opt':
        is_training_kwarg = {'train': is_training}
    else:
        is_training_kwarg = {'deterministic': not is_training}

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, **is_training_kwarg)[0]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(pred_rng, batch, params, model, generation_config):
    output_ids = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        generation_config=generation_config,
        params=params,
        prng_key=pred_rng)
    return output_ids.sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def main(dataset_name='xsum',
         text_key='document',
         model_name_or_path='gpt2-large',
         mesh_model_shards=2,
         n_epochs=2,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         max_length=512,
         learning_rate=4e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         top_p=0.96,
         jax_seed=42,
         workdir='./workdir',
         run_tensorboard=False):
    dataset = {
        'train': list(datasets.load_dataset(dataset_name, split='train')),
        'validation': [{text_key: ''} for _ in range(50)]
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    with jax.default_device(jax.devices('cpu')[0]):
        model = FlaxAutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.params = model.to_fp32(model.params)

    generation_config = GenerationConfig.from_pretrained(
        model_name_or_path,
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        pad_token_id=model.config.eos_token_id)

    deployer = Deployer(
        jax_seed=jax_seed,
        mesh_model_shards=mesh_model_shards,
        workdir=workdir,
        run_tensorboard=run_tensorboard,
        verbose=True)
    deployer.log_info(
        generation_config.to_json_string(), title='generation config')

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    params_shard_rules = deployer.guess_shard_rules(params=model.params)

    collate_fn_ = partial(collate_fn, tokenizer=tokenizer, text_key=text_key)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(collate_fn_, max_length=max_length),
        apply_fn=model.__call__,
        loss_fn=partial(loss_fn, model_type=model.config.model_type),
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_shard_rules=params_shard_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(collate_fn_, max_length=1),
        pred_fn=partial(
            pred_fn, model=model, generation_config=generation_config),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params=model.params,
        params_shard_rules=params_shard_rules)

    trainer.fit(
        train_examples=dataset['train'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=False,
        eval_predictor=predictor)


if __name__ == '__main__':
    fire.Fire(main)