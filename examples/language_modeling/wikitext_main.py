import os
from functools import partial
from itertools import chain
import fire
import datasets
import numpy as np
import jax
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from redco import Deployer, Trainer
from language_modeling_pipeline import loss_fn


def group_texts(examples, block_size):
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def collate_fn(examples):
    batch = {
        key: np.stack([example[key] for example in examples])
        for key in examples[0].keys()
    }
    batch['labels'] = batch['input_ids'][..., 1:]
    batch['input_ids'] = batch['input_ids'][..., :-1]
    batch['attention_mask'] = batch['attention_mask'][..., :-1]
    return batch


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
    dataset = {split: list(dataset[split]) for split in dataset.keys()}

    deployer = Deployer(
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        workdir=workdir,
        run_tensorboard=run_tensorboard,
        verbose=True)

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.__call__,
        loss_fn=partial(loss_fn, model_type=model.config.model_type),
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_shard_rules=deployer.guess_shard_rules(params=model.params))

    trainer.fit(
        train_examples=dataset['train'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True)


if __name__ == '__main__':
    fire.Fire(main)