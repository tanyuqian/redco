import os
from functools import partial
import fire
from PIL import Image
import numpy as np
import jax.numpy as jnp
import optax
import datasets
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoConfig,
    FlaxVisionEncoderDecoderModel)
import evaluate
from redco import Deployer, Trainer, Predictor


def collate_fn(examples,
               image_processor,
               tokenizer,
               decoder_start_token_id,
               max_tgt_len,
               image_path_key='image_path',
               text_key='caption'):
    model_inputs = {
        'pixel_values': image_processor([
            Image.open(example[image_path_key]).convert('RGB')
            for example in examples
        ], return_tensors='np').pixel_values
    }

    decoder_inputs = tokenizer(
        [example[text_key] for example in examples],
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


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop('labels')
    label_weights = batch['decoder_attention_mask']
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(pred_rng, batch, params, model):
    return model.generate(
        batch["pixel_values"], params=params, prng_key=pred_rng).sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def eval_rouge(examples, preds, text_key):
    rouge_scorer = evaluate.load('rouge')

    return rouge_scorer.compute(
        predictions=preds,
        references=[example[text_key] for example in examples],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(data_dir='./mscoco_data',
         image_path_key='image_path',
         text_key='caption',
         model_name_or_path='nlpconnect/vit-gpt2-image-captioning',
         init_ckpt_dir='./vit-gpt2-image-captioning',
         n_model_shards=1,
         max_tgt_len=16,
         num_beams=4,
         n_epochs=2,
         global_batch_size=32,
         per_device_batch_size=8,
         learning_rate=2e-5,
         lr_schedule_type='linear',
         warmup_rate=0.1,
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

    dataset = datasets.load_dataset(
        "ydshieh/coco_dataset_script", "2017",
        data_dir=os.path.abspath(f'{data_dir}/raw'),
        cache_dir=f'{data_dir}/cache')
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxVisionEncoderDecoderModel._from_config(
        config=AutoConfig.from_pretrained(model_name_or_path),
        dtype=jnp.bfloat16, _do_init=False)
    model.generation_config.update(
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_length=max_tgt_len,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=num_beams)

    accumulate_grad_batches = deployer.get_accumulate_grad_batches(
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size)
    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        schedule_type=lr_schedule_type,
        warmup_rate=warmup_rate)
    optimizer = optax.MultiSteps(optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adamw(learning_rate=lr_schedule_fn, weight_decay=weight_decay)
    ), every_k_schedule=accumulate_grad_batches)

    ckpt, info = deployer.load_last_ckpt(
        optimizer=optimizer, float_dtype=jnp.float32)
    if ckpt is None:
        ckpt, info = deployer.load_ckpt(
            ckpt_dir=init_ckpt_dir, update_rng=False, float_dtype=jnp.float32)

    params_sharding_rules = deployer.get_sharding_rules(
        params_shape_or_params=ckpt['params'])
    if params_sharding_rules is not None:
        deployer.log_info(
            info='\n'.join([str(t) for t in params_sharding_rules]),
            title='Sharding rules')

    collate_fn_kwargs = {
        'image_processor': image_processor,
        'tokenizer': tokenizer,
        'decoder_start_token_id': model.config.decoder_start_token_id,
        'max_tgt_len': max_tgt_len,
        'image_path_key': image_path_key,
        'text_key': text_key
    }
    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        apply_fn=model,
        loss_fn=loss_fn,
        params=ckpt['params'],
        opt_state=ckpt['opt_state'],
        last_ckpt_info=info,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=accumulate_grad_batches,
        params_sharding_rules=params_sharding_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        pred_fn=partial(pred_fn, model=model),
        output_fn=partial(output_fn, tokenizer=tokenizer),
        params_sharding_rules=params_sharding_rules)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['validation'],
        eval_predictor=predictor,
        eval_metric_fn=partial(eval_rouge, text_key=text_key),
        save_last_ckpt=False,
        save_argmax_ckpt_by_metrics=['rougeL'])


if __name__ == '__main__':
    fire.Fire(main)
