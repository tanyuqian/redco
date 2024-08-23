This is a MNIST example with RedCoast (`pip install redco==0.4.22`), supporting differentially-private (DP) training.
We provide utils for DP training, `dp_utils.py`, in the [DP training example](https://github.com/tanyuqian/redco/tree/master/examples/differential_private_training), which can be downloaded by
```
wget https://raw.githubusercontent.com/tanyuqian/redco/master/examples/differential_private_training/dp_utils.py
```

After downloading this, MNIST can be trained with data-privacy by 
applying a DP optimizer and a customized `train_step_fn` in `redco.Trainer`.  

```
python main.py --noise_multiplier 1.
```

To simulate multiple devices in cpu-only envs,
```
XLA_FLAGS="--xla_force_host_platform_device_count=8" python main.py --noise_multiplier 1.
```

### Source Code (`main.py`)
```python
from functools import partial
import fire
import numpy as np
from flax import linen as nn
import optax
from torchvision.datasets import MNIST
from redco import Deployer, Trainer, Predictor

import dp_utils

# A simple CNN model
# Copied from https://github.com/google/flax/blob/main/examples/mnist/train.py
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


# Collate function converting a batch of raw examples to model inputs (in numpy)
def collate_fn(examples):
    images = np.stack(
        [np.array(example['image'])[:, :, None] for example in examples])
    labels = np.array([example['label'] for example in examples])

    return {'images': images, 'labels': labels}


# Loss function converting model inputs to a scalar loss
def loss_fn(rng, state, params, batch, is_training):
    logits = state.apply_fn({'params': params}, batch['images'])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels']).mean()


# Predict function converting model inputs to the model outputs
def pred_fn(rng, params, batch, model):
    return model.apply({'params': params}, batch['images']).argmax(axis=-1)


# (Optional) Evaluation function in trainer.fit. Here it computes accuracy.
def eval_metric_fn(examples, preds):
    preds = np.array(preds)
    labels = np.array([example['label'] for example in examples])
    return {'acc': np.mean(preds == labels).item()}


def main(per_device_batch_size=64,
         learning_rate=1e-3,
         jax_seed=42,
         noise_multiplier=1.):
    deployer = Deployer(jax_seed=jax_seed, workdir='./workdir')

    dataset = {
        'train': [{'image': t[0], 'label': t[1]} for t in list(
            MNIST('./data', train=True, download=True))],
        'test': [{'image': t[0], 'label': t[1]} for t in list(
            MNIST('./data', train=False, download=True))],
    }

    model = CNN()
    dummy_batch = collate_fn(examples=[dataset['train'][0]])
    params = model.init(deployer.gen_rng(), dummy_batch['images'])['params']

    optimizer = optax.chain(
        optax.contrib.differentially_private_aggregate(
            l2_norm_clip=1.,
            noise_multiplier=noise_multiplier,
            seed=jax_seed),
        optax.adamw(learning_rate=learning_rate)
    )

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer,
        train_step_fn=dp_utils.dp_train_step)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=partial(pred_fn, model=model))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=2,
        eval_examples=dataset['test'],
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn)


if __name__ == '__main__':
    fire.Fire(main)
```