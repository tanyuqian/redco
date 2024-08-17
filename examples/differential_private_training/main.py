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
from dp_accounting import dp_event
from dp_accounting import rdp


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
        x = nn.Dense(features=64, dtype=jnp.float16)(x)
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
def loss_fn(train_rng, state, params, batch, is_training):
    logits = state.apply_fn({'params': params}, batch['images'])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels']).mean()


# Predict function converting model inputs to the model outputs
def pred_fn(pred_rng, params, batch, model):
    accs = model.apply({'params': params}, batch['images']).argmax(axis=-1)
    return {'acc': accs}


# (optional) Evaluation function in trainer.fit. Here it computes accuracy.
def eval_metric_fn(examples, preds):
    preds = np.array([pred['acc'] for pred in preds])
    labels = np.array([example['label'] for example in examples])
    return {'acc': np.mean(preds == labels).item()}


# Copied from
# https://github.com/google/jax/blob/main/examples/differentially_private_sgd.py
def compute_epsilon(
        steps, num_examples, batch_size, noise_multiplier, target_delta):
  if num_examples * target_delta > 1.:
    print('Your delta might be too high.')
  q = batch_size / float(num_examples)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
  accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)
  accountant.compose(
      dp_event.PoissonSampledDpEvent(
          q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
  return accountant.get_epsilon(target_delta)


# adapted from default_train_step()
# https://github.com/tanyuqian/redco/blob/master/redco/trainers/utils.py
def dp_train_step(train_rng,
                  state,
                  batch,
                  loss_fn,
                  lr_schedule_fn,
                  mesh,
                  compute_dtype):
    def loss_and_grads(batch_):
        return jax.value_and_grad(
            lambda params: loss_fn(
                train_rng=train_rng,
                state=state,
                params=params,
                batch=batch_,
                is_training=True)
        )(jax.tree.map(lambda x: x.astype(compute_dtype), state.params))

    def loss_and_per_sample_grads(batch_):
        batch_ = jax.tree.map(lambda x: x[:, None], batch_)
        loss, grads = jax.vmap(loss_and_grads)(batch_)

        return loss.mean(), grads

    if mesh is None:
        loss, grads = loss_and_per_sample_grads(batch)
        grads = jax.lax.pmean(grads, axis_name='dp')
    else:
        loss, grads = jax.vmap(loss_and_per_sample_grads)(batch)
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


def main(global_batch_size=256,
         per_device_batch_size=256,
         n_epochs=15,
         learning_rate=1e-3,
         noise_multiplier=1.,
         l2_norm_clip=1.5,
         delta=1e-5,
         jax_seed=42):
    deployer = Deployer(jax_seed=jax_seed, workdir='./workdir')

    dataset = {
        'train': [{'image': t[0], 'label': t[1]} for t in list(
            MNIST('./data', train=True, download=True))],
        'test': [{'image': t[0], 'label': t[1]} for t in list(
            MNIST('./data', train=False, download=True))],
    }
    print(len(dataset['train']), len(dataset['test']))

    model = CNN()
    dummy_batch = collate_fn(examples=[dataset['train'][0]])
    params = model.init(deployer.gen_rng(), dummy_batch['images'])['params']

    accumulate_grad_batches = deployer.get_accumulate_grad_batches(
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size)

    # optax.MultiSteps hasn't supported optax.differentially_private_aggregate
    assert accumulate_grad_batches == 1

    optimizer = optax.chain(
        optax.contrib.differentially_private_aggregate(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            seed=jax_seed),
        optax.adamw(learning_rate=learning_rate)
    )

    deployer.log_info(compute_epsilon(
        steps=n_epochs * len(dataset['train']) // global_batch_size,
        num_examples=len(dataset['train']),
        batch_size=global_batch_size,
        noise_multiplier=noise_multiplier,
        target_delta=delta
    ), title='Epsilon')

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer,
        accumulate_grad_batches=accumulate_grad_batches,
        train_step_fn=dp_train_step)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=partial(pred_fn, model=model))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['test'],
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn)


if __name__ == '__main__':
    fire.Fire(main)