from functools import partial
import fire
import numpy as np
import jax.numpy as jnp
import optax
import datasets
from transformers import AutoTokenizer, AutoConfig, FlaxAutoModelForSeq2SeqLM
import evaluate
from redco import Deployer, Trainer, Predictor


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


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop('labels')
    label_weights = batch['decoder_attention_mask']
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)
    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(pred_rng, params, batch, model):
    return model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=pred_rng).sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def eval_rouge(examples, preds, tgt_key):
    rouge_scorer = evaluate.load('rouge')
    return rouge_scorer.compute(
        predictions=preds,
        references=[example[tgt_key] for example in examples],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(dataset_name='EdinburghNLP/xsum',
         src_key='document',
         tgt_key='summary',
         model_name_or_path='google/flan-t5-xxl',
         init_ckpt_dir='./flan-t5-xxl',
         n_model_shards=8,
         max_src_len=512,
         max_tgt_len=64,
         num_beams=4,
         n_epochs=1,
         global_batch_size=64,
         per_device_batch_size=32,
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

    dataset = {
        split: list(datasets.load_dataset(dataset_name, split=split))
        for split in ['train', 'validation']
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    compute_dtype = jnp.bfloat16
    model = FlaxAutoModelForSeq2SeqLM.from_config(
        AutoConfig.from_pretrained(model_name_or_path),
        dtype=compute_dtype, _do_init=False)
    model.generation_config.update(
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_length=max_tgt_len,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=num_beams)

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
        loss_fn=loss_fn,
        params=ckpt.pop('params'),
        opt_state=ckpt.pop('opt_state'),
        last_ckpt_info=info,
        optimizer=optimizer,
        compute_dtype=compute_dtype,
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
        eval_metric_fn=partial(eval_rouge, tgt_key=tgt_key),
        save_last_ckpt=False,
        save_argmax_ckpt_by_metrics=['rougeL'])


if __name__ == '__main__':
    fire.Fire(main)
