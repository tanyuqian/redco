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
import jax
import jax.numpy as jnp
from jax.experimental.pjit import PartitionSpec as P
from jax_llama import convert_llama_weights, LLaMATokenizer, FlaxLLaMAForCausalLM

from redco import Deployer, Trainer, Predictor
from language_modeling_pipeline import collate_fn, loss_fn, pred_fn, output_fn


def main(dataset_name='xsum',
         text_key='document',
         llama_tokenizer_path='./llama_ckpt/tokenizer.model',
         llama_ckpt_dir='./llama_ckpt/7B',
         n_model_shards=4,
         n_epochs=2,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         max_length=32,
         learning_rate=1e-5,
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

    deployer = Deployer(
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        workdir=workdir,
        run_tensorboard=run_tensorboard,
        verbose=True)

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = LLaMATokenizer(llama_tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        params, configs = convert_llama_weights(llama_ckpt_dir, tokenizer)
        params = jax.tree_map(lambda x: jnp.asarray(x), params)
        model = FlaxLLaMAForCausalLM(configs, _do_init=False)
        params_shard_rules = deployer.guess_shard_rules(params=params)

        gen_kwargs = {
            'do_sample': True,
            'top_p': top_p,
            'max_length': max_length,
            'pad_token_id': model.config.eos_token_id
        }

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
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            text_key=text_key,
            max_length=max_length),
        apply_fn=model.__call__,
        loss_fn=partial(loss_fn, model_type=model.config.model_type),
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_shard_rules=params_shard_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn, tokenizer=tokenizer, text_key=text_key, max_length=1),
        pred_fn=partial(pred_fn, model=model, gen_kwargs=gen_kwargs),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params=params,
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