from functools import partial
import json
import fire
import numpy as np
import jax.numpy as jnp
import optax
import datasets
from transformers import AutoConfig, FlaxAutoModelForCausalLM, AutoTokenizer
from redco import Deployer, Predictor


def collate_fn(examples, tokenizer, max_src_len):
    texts = tokenizer.apply_chat_template([[{
        'role': 'user',
        'content': ' '.join([example['instruction'], example['input']])
    }] for example in examples], tokenize=False, add_generation_prompt=True)

    return tokenizer(
        texts,
        max_length=max_src_len,
        truncation=True,
        padding='max_length',
        add_special_tokens=False,
        return_tensors='np')


def pred_fn(rng, params, batch, model):
    return model.generate(**batch, params=params, prng_key=rng).sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def main(dataset_name='tatsu-lab/alpaca',
         model_name_or_path='LLM360/K2-Chat',
         ckpt_dir='./K2-Chat',
         max_src_len=128,
         max_tgt_len=128,
         top_p=0.95,
         n_model_shards=4,
         per_device_batch_size=32,
         workdir='./workdir',
         jax_seed=42,
         n_processes=None,
         host0_address=None,
         host0_port=None,
         process_id=None,
         n_local_devices=None):
    deployer = Deployer(
        workdir=workdir,
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        n_processes=n_processes,
        host0_address=host0_address,
        host0_port=host0_port,
        process_id=process_id,
        n_local_devices=n_local_devices)

    examples = list(datasets.load_dataset(dataset_name, split='train'))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = FlaxAutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(model_name_or_path),
        dtype=jnp.bfloat16, _do_init=False)
    model.generation_config.update(
        max_new_tokens=max_tgt_len, do_sample=True, top_p=top_p)

    ckpt, info = deployer.load_ckpt(
        ckpt_dir=ckpt_dir, float_dtype=jnp.bfloat16, load_opt_state=False)
    params_sharding_rules = deployer.get_sharding_rules(
        params_shape_or_params=ckpt['params'])
    if n_model_shards > 1:
        deployer.log_info(
            info='\n'.join([str(t) for t in params_sharding_rules]),
            title='Sharding rules')

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_src_len=max_src_len),
        pred_fn=partial(pred_fn, model=model),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params_sharding_rules=params_sharding_rules)

    preds = predictor.predict(
        examples,
        per_device_batch_size=per_device_batch_size,
        params=ckpt.pop('params'),
        desc=dataset_name.split('/')[-1])

    outputs = [
        {'example': example, 'pred': pred}
        for example, pred in zip(examples, preds)
    ]
    json.dump(outputs, open(f'{workdir}/outputs.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
