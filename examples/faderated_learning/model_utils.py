import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import optax


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
