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
from functools import partial
import fire
import datasets
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
from redco import Deployer, Predictor

from modeling_flax_mistral import FlaxMistralForCausalLM


def collate_fn(examples, tokenizer, src_length, src_key):
    batch = tokenizer(
        [example[src_key] for example in examples],
        max_length=src_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='np')

    return {
        key: batch[key] for key in ['input_ids', 'attention_mask']
    }


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


def main(n_processes=None,
         host0_address=None,
         host0_port=None,
         process_id=None,
         n_local_devices=None,
         dataset_name='tatsu-lab/alpaca',
         src_key='src',
         tgt_key='tgt',
         model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
         params_dir=None,
         computation_dtype='float32',
         per_device_batch_size=8,
         n_model_shards=1,
         max_length=512,
         eval_src_length=256,
         top_p=0.96,
         jax_seed=42,
         output_filename='gen_validation.json'):
    deployer = Deployer(
        n_model_shards=n_model_shards,
        jax_seed=jax_seed,
        n_processes=n_processes,
        host0_address=host0_address,
        host0_port=host0_port,
        process_id=process_id,
        n_local_devices=n_local_devices)

    # process Alpaca data as list
    # [{'src': ..., 'tgt': ...}, {'src': ..., 'tgt': ...}, ...]
    dataset = []
    for example in datasets.load_dataset(dataset_name, split='train'):
        dataset.append({
            src_key: example['text'][:-len(example['output'])].strip(),
            tgt_key: example['output']
        })
    train_size = int(0.9 * len(dataset))
    dataset = {
        'train': dataset[:train_size],
        'validation': dataset[train_size:],
    }

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        if 'mistral' in model_name_or_path.lower():
            model = FlaxMistralForCausalLM.from_pretrained(
                model_name_or_path,
                from_pt=True,
                dtype=getattr(jnp, computation_dtype))
        else:
            model = FlaxAutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                from_pt=True,
                dtype=getattr(jnp, computation_dtype))

        params = model.params if params_dir is None \
            else deployer.load_params(params_dir)

        # Sometimes params_dtype for inference can be fp16 for speed
        params = model.to_fp32(params)
        # params = model.to_fp16(params)

        gen_kwargs = {
            'do_sample': True,
            'top_p': top_p,
            'max_new_tokens': max_length - eval_src_length,
            'pad_token_id': tokenizer.pad_token_id
        }

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            src_length=eval_src_length,
            src_key=src_key),
        pred_fn=partial(pred_fn, model=model, gen_kwargs=gen_kwargs),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params_sharding_rules=deployer.get_sharding_rules(params=params))

    demo_src = \
        ('Below is an instruction that describes a task. Write a response that '
         'appropriately completes the request. '
         '### Instruction: Find the capital of Spain. ### Response:')
    demo_gen = predictor.predict(
        examples=[{src_key: demo_src}],
        params=params,
        per_device_batch_size=per_device_batch_size,
        desc='demo')[0]
    print(json.dumps({'demo': {'src': demo_src, 'gen': demo_gen}}, indent=4))

    outputs = predictor.predict(
        examples=dataset['validation'],
        params=params,
        per_device_batch_size=per_device_batch_size,
        desc='Validation set')
    gens = [
        {'example': example, 'generation': output}
        for example, output in zip(dataset['validation'], outputs)
    ]
    json.dump(gens, open(output_filename, 'w'), indent=4)
    print(f'Outputs saved in to {output_filename}.')


if __name__ == '__main__':
    fire.Fire(main)