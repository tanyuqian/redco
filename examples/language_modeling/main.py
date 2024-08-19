import fire
import numpy as np
import jax.numpy as jnp
import optax
import datasets
from transformers import AutoConfig, FlaxAutoModelForCausalLM
from redco import Deployer, Trainer


def get_dataset(dataset_name, eos_token_id, context_length):
    examples = []
    tokens_buffer = []
    for example in datasets.load_dataset(dataset_name, split='train'):
        tokens_buffer.extend(example['input_ids'][1:] + [eos_token_id])
        while len(tokens_buffer) >= context_length:
            examples.append({'token_ids': tokens_buffer[:context_length]})
            tokens_buffer = tokens_buffer[context_length:]

    cut = int(0.9 * len(examples))
    return {'train': examples[:cut], 'validation': examples[cut:]}


def collate_fn(examples):
    token_ids = np.array([examples['token_ids'] for examples in examples])
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop("labels")
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()


def main(dataset_name='alexgshaw/llama-13b-tokenized-wikitext-2-v1',
         model_name_or_path='huggyllama/llama-13b',
         init_ckpt_dir='./llama-13b',
         n_model_shards=8,
         n_epochs=1,
         global_batch_size=8,
         per_device_batch_size=4,
         learning_rate=2e-5,
         lr_schedule_type='linear',
         warmup_ratio=0.1,
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

    compute_dtype = jnp.bfloat16
    model = FlaxAutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(model_name_or_path),
        dtype=compute_dtype, _do_init=False)
    dataset = get_dataset(
        dataset_name=dataset_name,
        eos_token_id=model.config.eos_token_id,
        context_length=model.config.max_sequence_length)

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

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
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

    trainer.fit(
        train_examples=dataset['train'],
        eval_examples=dataset['validation'],
        n_epochs=n_epochs,
        per_device_batch_size=per_device_batch_size,
        save_every_ckpt=False,
        save_last_ckpt=False,
        save_opt_states=False)


if __name__ == '__main__':
    fire.Fire(main)