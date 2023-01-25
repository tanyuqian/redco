from functools import partial
import fire
import datasets
import numpy as np

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
import optax

from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

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


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop("labels")
    label_weights = batch['attention_mask']

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(pred_rng, batch, params, model, gen_kwargs):
    output_ids = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=pred_rng,
        **gen_kwargs)
    return output_ids.sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def main(dataset_name='xsum',
         text_key='document',
         model_name_or_path='facebook/opt-350m',
         mesh_model_shards=2,
         n_epochs=2,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         max_length=32,
         learning_rate=4e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         top_p=0.96,
         jax_seed=42):
    dataset = {
        'train': list(datasets.load_dataset(dataset_name, split='train')),
        'validation': [{text_key: ''} for _ in range(50)]
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    with jax.default_device(jax.devices('cpu')[0]):
        model = FlaxAutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.params = model.to_fp32(model.params)

    deployer = Deployer(jax_seed=jax_seed, mesh_model_shards=mesh_model_shards)

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    params_shard_rules = deployer.guess_shard_rules(params=model.params)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            text_key=text_key,
            max_length=max_length),
        apply_fn=model.__call__,
        loss_fn=loss_fn,
        params=freeze(model.params),
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_shard_rules=params_shard_rules)

    gen_kwargs = {
        'max_length': max_length,
        'do_sample': True,
        'top_p': top_p,
        'pad_token_id': tokenizer.eos_token_id
    }

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            text_key=text_key,
            max_length=1),
        pred_fn=partial(pred_fn, model=model, gen_kwargs=gen_kwargs),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params=freeze(model.params),
        params_shard_rules=params_shard_rules)

    trainer.fit(
        train_examples=dataset['train'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor)


if __name__ == '__main__':
    fire.Fire(main)