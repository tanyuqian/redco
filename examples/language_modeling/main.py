#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import partial
import fire
import datasets
import numpy as np
import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
from redco import Deployer, Trainer, Predictor

from modeling_flax_llama import FlaxLlamaForCausalLM


def train_collate_fn(examples, tokenizer, max_length, text_key, tgt_key):
    batch = tokenizer(
        [example[text_key] for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='np')

    batch['labels'] = np.copy(batch['input_ids'])
    batch['labels'][:, :-1] = batch['input_ids'][:, 1:]
    batch['labels'][:, -1] = tokenizer.eos_token_id

    is_tgt_token = np.zeros_like(batch['input_ids'])
    for i, example in enumerate(examples):
        tgt_ids = tokenizer(
            example[tgt_key], add_special_tokens=False)['input_ids']
        is_tgt_token[i, -len(tgt_ids):] = 1
    batch['label_weights'] = np.zeros_like(batch['input_ids'])
    batch['label_weights'][:, :-1] = is_tgt_token[:, 1:]
    batch['label_weights'][:, -1] = 1

    return {
        key: batch[key]
        for key in ['input_ids', 'attention_mask', 'labels', 'label_weights']
    }


def eval_collate_fn(examples, tokenizer, src_length, text_key, tgt_key):
    srcs = [
        example[text_key][:-len(example[tgt_key])].strip()
        for example in examples
    ]

    batch = tokenizer(
        srcs,
        max_length=src_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='np')

    return {
        key: batch[key] for key in ['input_ids', 'attention_mask']
    }


def loss_fn(train_rng, state, params, batch, is_training):
    labels, label_weights = batch.pop("labels"), batch.pop('label_weights')

    # logits = state.apply_fn(
    #     **batch, params=params, dropout_rng=train_rng, train=is_training)[0]
    logits = state.apply_fn(
        **batch,
        params=params,
        dropout_rng=train_rng,
        deterministic=not is_training)[0]

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


def main(dataset_name='tatsu-lab/alpaca',
         text_key='text',
         tgt_key='output',
         model_name_or_path='facebook/opt-350m',
         n_model_shards=2,
         n_epochs=3,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         max_length=64,
         eval_src_length=64,
         learning_rate=2e-5,
         lr_schedule_type='cosine',
         warmup_rate=0.03,
         weight_decay=0.,
         top_p=0.96,
         jax_seed=42,
         workdir='./workdir',
         run_tensorboard=False):
    dataset = list(datasets.load_dataset(dataset_name, split='train'))
    train_size = int(0.9 * len(dataset))
    dataset = {
        'train': dataset[:train_size],
        'validation': dataset[train_size:],
    }

    deployer = Deployer(
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        workdir=workdir,
        run_tensorboard=run_tensorboard,
        verbose=True)

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_size='right')
        tokenizer.pad_token = tokenizer.eos_token

        if 'llama' in model_name_or_path.lower():
            model = FlaxLlamaForCausalLM.from_pretrained(
                model_name_or_path, from_pt=True)
        else:
            model = FlaxAutoModelForCausalLM.from_pretrained(model_name_or_path)

        params = model.to_fp32(model.params)

        gen_kwargs = {
            'do_sample': True,
            'top_p': top_p,
            'max_new_tokens': max_length - eval_src_length,
            'pad_token_id': tokenizer.pad_token_id
        }

    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        schedule_type=lr_schedule_type,
        warmup_rate=warmup_rate)

    optimizer = optax.adamw(
        learning_rate=lr_schedule_fn, weight_decay=weight_decay)
    if accumulate_grad_batches > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=accumulate_grad_batches)

    params_sharding_rules = deployer.get_sharding_rules(params=params)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            train_collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            text_key=text_key,
            tgt_key=tgt_key),
        apply_fn=model,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_sharding_rules=params_sharding_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            eval_collate_fn,
            tokenizer=tokenizer,
            src_length=eval_src_length,
            text_key=text_key,
            tgt_key=tgt_key),
        pred_fn=partial(pred_fn, model=model, gen_kwargs=gen_kwargs),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params_sharding_rules=params_sharding_rules)

    trainer.fit(
        train_examples=dataset['train'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_predictor=predictor)


if __name__ == '__main__':
    fire.Fire(main)