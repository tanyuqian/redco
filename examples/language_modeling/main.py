import os
from functools import partial
from itertools import chain
import fire
import datasets
import numpy as np

import jax
import jax.numpy as jnp
import optax

from transformers import \
    AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig

from redco import Deployer, Trainer, Predictor


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported
    # it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def collate_fn(examples):
    batch = {
        key: np.stack([example[key] for example in examples])
        for key in examples[0].keys()
    }
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


def main(text_key='text',
         model_name_or_path='gpt2-large',
         n_model_shards=2,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    with jax.default_device(jax.devices('cpu')[0]):
        model = FlaxAutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.params = model.to_fp32(model.params)

    raw_dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenized_dataset = raw_dataset.map(
        lambda example: tokenizer(example[text_key]),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=list(raw_dataset['train'][0].keys()),
        load_from_cache_file=True,
        desc="Running tokenizer on dataset")
    dataset = tokenized_dataset.map(
        partial(group_texts, block_size=max_length),
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {max_length}")

    try:
        generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    except:
        generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        pad_token_id=model.config.eos_token_id)

    deployer = Deployer(
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
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

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.__call__,
        loss_fn=partial(loss_fn, model_type=model.config.model_type),
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_shard_rules=params_shard_rules)

    trainer.fit(
        train_examples=dataset['train'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True)


if __name__ == '__main__':
    fire.Fire(main)