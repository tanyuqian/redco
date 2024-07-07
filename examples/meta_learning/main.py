from functools import partial
import fire
import tqdm
import numpy as np
import jax
import flax.linen as nn
import optax
import learn2learn as l2l
from redco import Deployer, Trainer, Predictor


class CNN(nn.Module):
    """
    A simple CNN model.
    Copied from https://github.com/google/flax/blob/main/examples/mnist/train.py
    """
    n_labels: int = 1

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
        x = nn.Dense(features=self.n_labels)(x)
        return x


def preprocess(l2l_example):
    return {
        'train': {
            'pixel_values': l2l_example[0][::2].numpy().transpose((0, 2, 3, 1)),
            'labels': l2l_example[1][::2].numpy()
        },
        'val': {
            'pixel_values': l2l_example[0][1::2].numpy().transpose(
                (0, 2, 3, 1)),
            'labels': l2l_example[1][1::2].numpy()
        },
    }


def collate_fn(examples):
    return {
        split: {
            key: np.stack([example[split][key] for example in examples])
            for key in examples[0][split].keys()
        } for split in ['train', 'val']
    }


def inner_loss_fn(params, batch, model):
    logits = model.apply({'params': params}, batch['pixel_values'])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels']).mean()


def inner_step(params, model, inner_batch, inner_learning_rate, inner_n_steps):
    inner_optimizer = optax.sgd(learning_rate=inner_learning_rate)
    inner_opt_state = inner_optimizer.init(params)

    for _ in range(inner_n_steps):
        grads = jax.grad(inner_loss_fn)(params, inner_batch, model)
        updates, inner_opt_state = inner_optimizer.update(
            updates=grads, state=inner_opt_state, params=params)
        params = optax.apply_updates(params, updates)

    return params


def loss_fn(train_rng,
            state,
            params,
            batch,
            is_training,
            model,
            inner_learning_rate,
            inner_n_steps):
    def inner_maml_loss_fn(inner_batch_train, inner_batch_val):
        params_upd = inner_step(
            params=params,
            model=model,
            inner_batch=inner_batch_train,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps)

        return inner_loss_fn(
            params=params_upd, batch=inner_batch_val, model=model)

    return jax.vmap(inner_maml_loss_fn)(batch['train'], batch['val']).mean()


def pred_fn(pred_rng,
            batch,
            params,
            model,
            inner_learning_rate,
            inner_n_steps):
    def inner_maml_pred_fn(inner_batch_train, inner_batch_val):
        params_upd = inner_step(
            params=params,
            model=model,
            inner_batch=inner_batch_train,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps)

        return model.apply(
            {'params': params_upd}, inner_batch_val['pixel_values']
        ).argmax(axis=-1)

    return jax.vmap(inner_maml_pred_fn)(batch['train'], batch['val'])


def eval_metric_fn(preds, examples):
    preds = np.array(preds)
    labels = np.array([example['val']['labels'] for example in examples])
    return {'acc': np.mean(preds == labels).item()}


def main(dataset_name='omniglot',
         n_ways=5,
         n_shots=1,
         n_tasks_per_epoch=1024,
         n_epochs=1000,
         per_device_batch_size=32,
         learning_rate=0.003,
         inner_learning_rate=0.5,
         inner_n_steps=1,
         jax_seed=42):
    deployer = Deployer(jax_seed=jax_seed, n_model_shards=1)

    taskset = l2l.vision.benchmarks.get_tasksets(
        name=dataset_name,
        train_ways=n_ways,
        train_samples=2 * n_shots,
        test_ways=n_ways,
        test_samples=2 * n_shots,
        root='./data')

    model = CNN(n_labels=n_ways)
    dummy_inputs = taskset.train.sample()[0].numpy().transpose((0, 2, 3, 1))
    params = model.init(deployer.gen_rng(), dummy_inputs)['params']
    optimizer = optax.adamw(learning_rate=learning_rate)

    params_sharding_rules = deployer.get_sharding_rules(params)
    if params_sharding_rules is not None:
        deployer.log_info(
            info='\n'.join([str(t) for t in params_sharding_rules]),
            title='Sharding rules')

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=partial(
            loss_fn,
            model=model,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps),
        params=params,
        optimizer=optimizer,
        params_sharding_rules=params_sharding_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=partial(
            pred_fn,
            model=model,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps),
        params_sharding_rules=params_sharding_rules)

    eval_examples = [
        preprocess(taskset.validation.sample())
        for _ in tqdm.trange(n_tasks_per_epoch, desc='Sampling eval tasks')
    ]
    train_examples_fn = lambda epoch_idx: [
        preprocess(taskset.train.sample())
        for _ in tqdm.trange(n_tasks_per_epoch, desc=f'Sampling training tasks')
    ]
    trainer.fit(
        train_examples=train_examples_fn,
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=eval_examples,
        eval_loss=False,
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn)


if __name__ == '__main__':
    fire.Fire(main)
