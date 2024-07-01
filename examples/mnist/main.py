from functools import partial
import fire
import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
import optax
from torchvision.datasets import MNIST
from redco import Deployer, Trainer, Predictor


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, training):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dropout(rate=0.4, deterministic=not training)(x)
        x = nn.Dense(features=64, dtype=jnp.float16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def collate_fn(examples):
    images = np.stack([
        np.expand_dims(np.array(example['image']), axis=-1)
        for example in examples])

    labels = np.array([example['label'] for example in examples])

    return {'images': images, 'labels': labels}


def loss_fn(train_rng, state, params, batch, is_training):
    logits = state.apply_fn(
        {'params': params}, batch['images'],
        training=is_training, rngs={'dropout': train_rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels'])
    return jnp.mean(loss)


def pred_fn(pred_rng, params, batch, model):
    accs = model.apply(
        {'params': params}, batch['images'], training=False).argmax(axis=-1)
    return {'acc': accs}


def eval_metric_fn(examples, preds):
    preds = np.array([pred['acc'] for pred in preds])
    labels = np.array([example['label'] for example in examples])
    return {'acc': np.mean(preds == labels).item()}


def main(data_dir='./data/',
         per_device_batch_size=64,
         n_model_shards=1,
         n_epochs=20,
         learning_rate=1e-3,
         jax_seed=42):
    dataset = {
        'train': [{'image': t[0], 'label': t[1]} for t in list(
            MNIST(data_dir, train=True, download=True))][:10000],
        'test': [{'image': t[0], 'label': t[1]} for t in list(
            MNIST(data_dir, train=False, download=True))],
    }

    deployer = Deployer(
        jax_seed=jax_seed,
        workdir='./workdir',
        run_tensorboard=False,
        wandb_init_kwargs=None,
        n_model_shards=n_model_shards)

    model = CNN()
    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        schedule_type='linear',
        warmup_rate=0.1)

    optimizer = optax.adamw(learning_rate=lr_schedule_fn, weight_decay=0.01)
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=2)

    ckpt, last_ckpt_info = deployer.load_last_ckpt(
        optimizer=optimizer,
        float_dtype=jnp.float32)
    if ckpt is None:
        dummy_batch = collate_fn([dataset['train'][0]])
        ckpt = {
            'params': model.init(deployer.gen_rng(), dummy_batch['images'], training=False)['params'],
            'opt_state': None
        }

    from flax.traverse_util import flatten_dict
    for key, value in flatten_dict(params).items():
        print(key, value.shape, value.dtype)

    params_sharding_rules = deployer.get_sharding_rules(
        params_shape_or_params=params)
    if params_sharding_rules is not None:
        deployer.log_info(
            info='\n'.join([str(t) for t in params_sharding_rules]),
            title='Sharding rules')

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        opt_state=opt_state,
        last_ckpt_info=last_ckpt_info,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        accumulate_grad_batches=2,
        params_sharding_rules=params_sharding_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=partial(pred_fn, model=model),
        params_sharding_rules=params_sharding_rules)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['test'],
        eval_per_device_batch_size=per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn,
        eval_sanity_check=True,
        save_every_ckpt=True,
        save_opt_states=True,
        save_float_dtype=jnp.float16,
        save_argmin_ckpt_by_metrics=['loss'])


if __name__ == '__main__':
    fire.Fire(main)
