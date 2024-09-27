from functools import partial
import fire
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import path_aware_map
from flax.core.frozen_dict import freeze
import optax
import datasets
from transformers import AutoTokenizer, AutoConfig, FlaxAutoModelForSeq2SeqLM
from redco import Deployer, Trainer, Predictor

from lora_utils import get_lora, aggregate_params


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


def collate_fn(examples,
               tokenizer,
               decoder_start_token_id,
               max_src_len,
               max_tgt_len,
               src_key='src',
               tgt_key='tgt'):
    model_inputs = tokenizer(
        [example[src_key] for example in examples],
        max_length=max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    decoder_inputs = tokenizer(
        [example[tgt_key] for example in examples],
        max_length=max_tgt_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    labels = decoder_inputs['input_ids']
    decoder_inputs['input_ids'] = np.zeros_like(labels)
    decoder_inputs['input_ids'][:, 0] = decoder_start_token_id
    decoder_inputs['input_ids'][:, 1:] = labels[:, :-1]

    model_inputs['labels'] = labels
    for key in decoder_inputs:
        model_inputs[f'decoder_{key}'] = np.array(decoder_inputs[key])

    return model_inputs


def loss_fn(rng, state, params, batch, is_training, lora_alpha):
    labels = batch.pop('labels')
    label_weights = batch['decoder_attention_mask']

    params = aggregate_params(
        model_params=params['model'],
        lora_params=params['lora'],
        lora_alpha=lora_alpha)

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=rng, train=is_training)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


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
         lora_rank=8,
         lora_alpha=32,
         init_ckpt_dir='./bart-base',
         n_model_shards=1,
         max_src_len=128,
         max_tgt_len=4,
         num_beams=1,
         n_epochs=1,
         global_batch_size=16,
         per_device_batch_size=16,
         learning_rate=1e-4,
         lr_schedule_type='linear',
         warmup_ratio=0.,
         grad_norm_clip=1.,
         weight_decay=0.01,
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
    lora_params = get_lora(
        model_params=model_init_ckpt['params'],
        lora_rank=lora_rank,
        rng=deployer.gen_rng())
    params = {'model': model_init_ckpt.pop('params'), 'lora': lora_params}

    params_sharding_rules = {
        'model': deployer.get_sharding_rules(params['model']),
        'lora': None
    }
    if n_model_shards > 1:
        deployer.log_info(
            info='\n'.join([str(t) for t in params_sharding_rules['model']]),
            title='Sharding rules')

    accumulate_grad_batches = deployer.get_accumulate_grad_batches(
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size)
    deployer.log_info(accumulate_grad_batches, title='accumulate_grad_batches')
    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        schedule_type=lr_schedule_type,
        warmup_ratio=warmup_ratio)
    optimizer = optax.MultiSteps(optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adamw(learning_rate=lr_schedule_fn, weight_decay=weight_decay)
    ), every_k_schedule=accumulate_grad_batches)
    optimizer = optax.multi_transform(
        transforms={'model': optax.set_to_zero(), 'lora': optimizer},
        param_labels=freeze(path_aware_map(lambda path, _: path[0], params))
    )

    collate_fn_kwargs = {
        'tokenizer': tokenizer,
        'decoder_start_token_id': model.config.decoder_start_token_id,
        'max_src_len': max_src_len,
        'max_tgt_len': max_tgt_len,
        'src_key': src_key,
        'tgt_key': tgt_key,
    }
    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        apply_fn=model,
        loss_fn=partial(loss_fn, lora_alpha=lora_alpha),
        params=params,
        optimizer=optimizer,
        compute_dtype=compute_dtype,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=accumulate_grad_batches,
        params_sharding_rules=params_sharding_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        pred_fn=partial(pred_fn, model=model, lora_alpha=lora_alpha),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params_sharding_rules=params_sharding_rules)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['validation'],
        eval_predictor=predictor,
        eval_metric_fn=partial(eval_acc, tgt_key=tgt_key))

    deployer.save_ckpt(
        './lora',
        params=trainer.get_params_to_save()['lora'],
        float_dtype=jnp.float32)


if __name__ == '__main__':
    fire.Fire(main)
