from functools import partial
import fire
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
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def collate_fn(examples):
    images, labels = [], []
    for image, label in examples:
        images.append(np.expand_dims(np.array(image), axis=(0, -1)))
        labels.append(label)

    return {
        'images': np.concatenate(images, axis=0),
        'labels': np.array(labels)
    }


def loss_fn(train_rng, state, params, batch, is_training):
    logits = state.apply_fn(
        {'params': params}, batch['images'],
        training=is_training, rngs={'dropout': train_rng})
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels'])
    return jnp.mean(loss)


def pred_fn(pred_rng, batch, params, model):
    return model.apply(
        {'params': params}, batch['images'], training=False).argmax(axis=-1)


def eval_metric_fn(eval_results):
    preds = np.array([result['pred'] for result in eval_results])
    labels = np.array([result['example'][1] for result in eval_results])
    return {'acc': np.mean(preds == labels).item()}


def main(data_dir='./data/',
         per_device_batch_size=64,
         n_epochs=2,
         learning_rate=1e-3,
         jax_seed=42):
    dataset = {
        'train': list(MNIST(data_dir, train=True, download=True)),
        'test': list(MNIST(data_dir, train=False, download=True))
    }

    deployer = Deployer(jax_seed=jax_seed)

    model = CNN()
    dummy_batch = collate_fn([dataset['train'][0]])
    params = model.init(
        deployer.gen_rng(), dummy_batch['images'], training=False)['params']
    optimizer = optax.adam(learning_rate=learning_rate)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=partial(pred_fn, model=model))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['test'],
        eval_per_device_batch_size=per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn)


if __name__ == '__main__':
    fire.Fire(main)
