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

import json
import fire
import numpy as np
import jax
import jax.numpy as jnp
import optax
from transformers import FlaxAutoModelForCausalLM
from redco import Deployer, Trainer


def collate_fn(examples):
    token_ids = np.array([examples['token_ids'] for examples in examples])
    token_ids[token_ids >= 32000] = 2
    return {'token_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop("labels")
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]

    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()


def main(n_processes=None,
         host0_address=None,
         host0_port=None,
         process_id=None,
         n_local_devices=None,
         data_file='chunk_379.jsonl',
         model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
         n_model_shards=1,
         n_epochs=1,
         global_batch_size=8,
         per_device_batch_size=1,
         learning_rate=2e-5,
         lr_schedule_type='linear',
         warmup_rate=0.03,
         weight_decay=0.,
         jax_seed=42,
         workdir='./workdir'):
    deployer = Deployer(
        n_model_shards=n_model_shards,
        jax_seed=jax_seed,
        workdir=workdir,
        n_processes=n_processes,
        host0_address=host0_address,
        host0_port=host0_port,
        process_id=process_id,
        n_local_devices=n_local_devices)

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_name_or_path, from_pt=True, dtype=jnp.bfloat16)
        params = model.to_fp32(model.params)

    dataset = {'train': [json.loads(line) for line in open(data_file)]}

    global_micro_batch_size, _ = deployer.process_batch_size(
        per_device_batch_size=per_device_batch_size)
    assert global_batch_size % global_micro_batch_size == 0
    accumulate_grad_batches = global_batch_size // global_micro_batch_size

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

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=accumulate_grad_batches,
        params_sharding_rules=deployer.get_sharding_rules(
            params_shape_or_params=params))

    trainer.fit(
        train_examples=dataset['train'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size)


if __name__ == '__main__':
    fire.Fire(main)