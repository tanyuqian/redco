import json
from functools import partial
import fire
import jax.numpy as jnp
import datasets
from transformers import AutoTokenizer, AutoConfig, FlaxAutoModelForSeq2SeqLM
from redco import Deployer, Predictor

from lora_utils import aggregate_params


def get_dataset():
    dataset = datasets.load_dataset("financial_phrasebank", "sentences_allagree")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=11111)
    dataset["validation"] = dataset["test"]
    del dataset["test"]
    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True)
    dataset = {split: list(dataset[split]) for split in ['train', 'validation']}
    src_key, tgt_key = 'sentence', 'text_label'
    print('DATA EXAMPLE:', dataset['train'][0])

    return dataset, src_key, tgt_key


def collate_fn(examples, tokenizer, max_src_len, src_key='src'):
    return tokenizer(
        [example[src_key] for example in examples],
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')


def pred_fn(rng, params, batch, model, lora_alpha):
    params = aggregate_params(
        model_params=params['model'],
        lora_params=params['lora'],
        lora_alpha=lora_alpha)

    return model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=rng).sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def eval_acc(examples, preds, tgt_key):
    n_correct = sum([
        int(pred == example[tgt_key]) for pred, example in zip(preds, examples)
    ])
    return {'acc': n_correct / len(examples)}



def main(model_name_or_path='facebook/bart-base',
         lora_alpha=32,
         init_ckpt_dir='./bart-base',
         lora_ckpt_dir='./lora',
         n_model_shards=1,
         max_src_len=128,
         max_tgt_len=4,
         num_beams=1,
         per_device_batch_size=8,
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

    dataset, src_key, tgt_key = get_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    compute_dtype = jnp.float16
    model = FlaxAutoModelForSeq2SeqLM.from_config(
        AutoConfig.from_pretrained(model_name_or_path),
        dtype=compute_dtype, _do_init=False)
    model.generation_config.update(
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_length=max_tgt_len,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=num_beams)

    model_init_ckpt, _ = deployer.load_ckpt(
        ckpt_dir=init_ckpt_dir, update_rng=False, float_dtype=jnp.float16)
    lora_ckpt, _ = deployer.load_ckpt(
        ckpt_dir=lora_ckpt_dir, float_dtype=jnp.float32)
    params = {
        'model': model_init_ckpt.pop('params'), 'lora': lora_ckpt.pop('params')
    }

    params_sharding_rules = {
        'model': deployer.get_sharding_rules(params['model']),
        'lora': None
    }
    if n_model_shards > 1:
        deployer.log_info(
            info='\n'.join([str(t) for t in params_sharding_rules['model']]),
            title='Sharding rules')

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_src_len=max_src_len,
            src_key=src_key),
        pred_fn=partial(pred_fn, model=model, lora_alpha=lora_alpha),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params_sharding_rules=params_sharding_rules)

    preds = predictor.predict(
        examples=dataset['validation'],
        per_device_batch_size=per_device_batch_size,
        params=params)

    for pred, example in zip(preds[:5], dataset['validation'][:5]):
        print(json.dumps({'example': example, 'pred': pred}, indent=4))

    print(eval_acc(
        preds=preds, examples=dataset['validation'], tgt_key=tgt_key))


if __name__ == '__main__':
    fire.Fire(main)
