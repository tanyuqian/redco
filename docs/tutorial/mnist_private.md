This is a MNIST example with RedCoast (`pip install redco==0.4.22`), supporting differentially-private training 
```
python main.py --noise_multiplier 1.
```

To simulate multiple devices in cpu-only envs,
```
XLA_FLAGS="--xla_force_host_platform_device_count=8" python main.py --noise_multiplier 1.
```

### Source Code
```python
from functools import partial
import fire
import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import l2_norm
from flax import linen as nn
import optax
from torchvision.datasets import MNIST
from redco import Deployer, Trainer, Predictor


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


# adapted from default_train_step(), added `loss_and_per_sample_grads`
# https://github.com/tanyuqian/redco/blob/master/redco/trainers/utils.py
def dp_train_step(
        rng, state, batch, loss_fn, lr_schedule_fn, mesh, compute_dtype):
    def loss_and_grads(rng_, batch_):
        return jax.value_and_grad(
            lambda params: loss_fn(
                rng=rng_,
                state=state,
                params=params,
                batch=batch_,
                is_training=True)
        )(jax.tree.map(lambda x: x.astype(compute_dtype), state.params))

    def loss_and_per_sample_grads(rng_, batch_):
        batch_ = jax.tree.map(lambda x: x[:, None], batch_)
        loss, grads = jax.vmap(lambda b: loss_and_grads(rng_, b))(batch_)

        return loss.mean(), grads

    if mesh is None:
        loss, grads = loss_and_per_sample_grads(rng, batch)
        grads = jax.lax.pmean(grads, axis_name='dp')
    else:
        loss, grads = jax.vmap(loss_and_per_sample_grads)(
            jax.random.split(rng, num=mesh.shape['dp']), batch)
        loss = jnp.mean(loss)
        grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)

    new_state = state.apply_gradients(grads=jax.tree.map(
        lambda grad, param: grad.astype(param.dtype), grads, state.params))

    metrics = {'loss': loss, 'step': state.step, 'grad_norm': l2_norm(grads)}
    if lr_schedule_fn is not None:
        metrics['lr'] = lr_schedule_fn(state.step)
    if mesh is None:
        metrics = jax.lax.pmean(metrics, axis_name='dp')

    return new_state, metrics


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
        train_step_fn=dp_train_step)

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