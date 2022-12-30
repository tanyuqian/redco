from functools import partial

import fire
import jax.random
import numpy as np

import jax.numpy as jnp
from flax import linen as nn
import optax

from torchvision.datasets import MNIST

from redco import Deployer, Trainer, Predictor


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
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


def loss_fn(state, params, batch, train):
    logits = state.apply_fn({'params': params}, batch['images'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels'])
    return jnp.mean(loss)


def pred_fn(batch, params, model):
    return model.apply({'params': params}, batch['images']).argmax(axis=-1)


def costum_eval_fn(params, examples, trainer, predictor, per_device_batch_size):
    preds = predictor.predict(
        params=params,
        examples=examples,
        per_device_batch_size=per_device_batch_size)

    labels = [example[1] for example in examples]

    loss = trainer.eval_loss(
        examples=examples, per_device_batch_size=per_device_batch_size)['loss']

    assert len(preds) == len(labels)

    return {'loss': loss, 'acc': np.mean(np.array(preds) == np.array(labels))}


def main(data_dir='./data/'):
    dataset = {
        'train': list(MNIST(data_dir, train=True, download=True)),
        'test': list(MNIST(data_dir, train=False, download=True))
    }

    deployer = Deployer(jax_seed=42)

    model = CNN()
    params = model.init(deployer.gen_rng(), np.ones([1, 28, 28, 1]))['params']
    optimizer = optax.adam(learning_rate=1e-3)

    trainer = Trainer(
        apply_fn=model.apply,
        params=params,
        optimizer=optimizer,
        deployer=deployer)

    trainer.setup(loss_fn=loss_fn, collate_fn=collate_fn)

    predictor = Predictor(model=model, deployer=deployer)

    predictor.setup(
        collate_fn=collate_fn, pred_fn=pred_fn, output_fn=lambda x: x.tolist())

    eval_fn = partial(
        costum_eval_fn,
        examples=dataset['test'],
        trainer=trainer,
        predictor=predictor,
        per_device_batch_size=64)

    trainer.fit(
        train_examples=dataset['train'],
        eval_examples=dataset['test'],
        n_epochs=2,
        per_device_batch_size=32,
        eval_fn=eval_fn)


if __name__ == '__main__':
    fire.Fire(main)
