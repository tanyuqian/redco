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
from jax_llama import \
    convert_llama_weights, LLaMATokenizer, FlaxLLaMAForCausalLM
from redco import Deployer, Trainer, Predictor


def collate_fn(examples,
               tokenizer,
               text_key,
               tgt_key,
               max_src_len,
               max_tgt_len,
               for_training):
    if for_training:
        texts = [example[text_key] for example in examples]
        max_length = max_src_len + max_tgt_len
    else:
        texts = []
        for example in examples:
            assert example[text_key].endswith(example[tgt_key])
            texts.append(example[text_key][:-len(example[tgt_key])])
        max_length = max_src_len

    batch = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='np')

    if for_training:
        batch['labels'] = np.copy(batch['input_ids'])
        batch['labels'][:, :-1] = batch['input_ids'][:, 1:]

    return batch


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop("labels")
    label_weights = batch['attention_mask']

    is_training_kwarg = {'train': is_training}
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, **is_training_kwarg)[0]

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
         llama_tokenizer_path='./llama_ckpt/tokenizer.model',
         llama_ckpt_dir='./llama_ckpt/7B',
         n_model_shards=4,
         n_epochs=2,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         max_src_len=512,
         max_tgt_len=512,
         learning_rate=1e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         top_p=0.96,
         jax_seed=42,
         workdir='./workdir',
         run_tensorboard=False):
    dataset = list(datasets.load_dataset(dataset_name, split='train'))
    dataset = {
        'train': dataset[:int(0.9 * len(dataset))],
        'validation': dataset[int(0.9 * len(dataset)):],
    }

    deployer = Deployer(
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        workdir=workdir,
        run_tensorboard=run_tensorboard,
        verbose=True)

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = LLaMATokenizer(llama_tokenizer_path)
        tokenizer.pad_token_id = 0
        assert tokenizer.pad_token_id != tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        params, configs = convert_llama_weights(llama_ckpt_dir, tokenizer)
        params = jax.tree_map(lambda x: jnp.asarray(x), params)
        model = FlaxLLaMAForCausalLM(configs, _do_init=False)
        params = model.to_fp32(params=params)

        gen_kwargs = {
            'do_sample': True,
            'top_p': top_p,
            'max_new_tokens': max_tgt_len,
            'pad_token_id': tokenizer.pad_token_id
        }

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    params_shard_rules = deployer.get_sharding_rules(params=params)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            text_key=text_key,
            tgt_key=tgt_key,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            for_training=True),
        apply_fn=model.__call__,
        loss_fn=partial(loss_fn, model_type=model.config.model_type),
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        params_shard_rules=params_shard_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            text_key=text_key,
            tgt_key=tgt_key,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            for_training=False),
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
        eval_loss=True,
        eval_predictor=predictor)


if __name__ == '__main__':
    fire.Fire(main)