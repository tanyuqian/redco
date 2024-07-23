from functools import partial
import fire
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, FlaxAutoModelForSequenceClassification)
from redco import Deployer, Trainer, Predictor


def collate_fn(
        examples, sent0_key, sent1_key, label_key, tokenizer, max_length):
    texts = []
    for example in examples:
        if sent1_key is None:
            texts.append(example[sent0_key])
        else:
            texts.append((example[sent0_key], example[sent1_key]))

    batch = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')
    batch['labels'] = np.array([example[label_key] for example in examples])

    return batch


def loss_fn(train_rng, state, params, batch, is_training, is_regression):
    labels = batch.pop('labels')
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training).logits

    if is_regression:
        return jnp.mean(jnp.square(logits[..., 0] - labels))
    else:
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels).mean()


def pred_fn(pred_rng, params, batch, model, is_regression):
    batch.pop('labels')

    logits = model(**batch, params=params, train=False).logits

    if is_regression:
        return logits[..., 0]
    else:
        return logits.argmax(axis=-1)


def eval_metric_fn(examples, preds, label_key, is_regression):
    preds = np.array(preds)
    labels = np.array([example[label_key] for example in examples])

    if is_regression:
        return {'square error': np.mean(np.square(preds - labels))}
    else:
        return {'acc': np.mean(preds == labels).item()}


def main(dataset_name='sst2',
         sent0_key='sentence',
         sent1_key=None,
         label_key='label',
         is_regression=False,
         model_name_or_path='FacebookAI/roberta-large',
         init_ckpt_dir='./roberta-large',
         max_length=512,
         n_model_shards=1,
         global_batch_size=32,
         per_device_batch_size=4,
         n_epochs=2,
         learning_rate=2e-5,
         warmup_rate=0.1,
         lr_schedule_type='linear',
         grad_norm_clip=1.,
         weight_decay=0.,
         jax_seed=42,
         workdir='./workdir',
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

    dataset = load_dataset('glue', dataset_name)
    dataset = {key: list(dataset[key]) for key in dataset.keys()}
    num_labels = 1 if is_regression \
        else len(set([example[label_key] for example in dataset['train']]))
    deployer.log_info(num_labels, title='num_labels')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    compute_dtype = jnp.bfloat16
    model = FlaxAutoModelForSequenceClassification.from_config(
        AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels),
        dtype=compute_dtype, _do_init=False)

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
        'sent0_key': sent0_key,
        'sent1_key': sent1_key,
        'label_key': label_key,
        'tokenizer': tokenizer,
        'max_length': max_length,
    }
    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        apply_fn=model,
        loss_fn=partial(loss_fn, is_regression=is_regression),
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
        pred_fn=partial(pred_fn, model=model, is_regression=is_regression),
        params_sharding_rules=params_sharding_rules)

    trainer.fit(
        train_examples=dataset['train'],
        eval_examples=dataset['validation'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_predictor=predictor,
        eval_metric_fn=partial(
            eval_metric_fn,
            label_key=label_key,
            is_regression=is_regression)
    )


if __name__ == '__main__':
    fire.Fire(main)
