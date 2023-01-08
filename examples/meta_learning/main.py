from functools import partial
import fire
import jax
import jax.numpy as jnp
import optax
import numpy as np

from redco import Deployer, Trainer, Predictor

from utils import CNN, get_torchmeta_dataset, sample_tasks


def collate_fn(examples):
    batch = {}
    for split in examples[0].keys():
        batch[split] = {
            key: np.stack([example[split][key] for example in examples])
            for key in examples[0][split].keys()
        }

    return batch


def task_loss_fn(apply_fn, params, task_split_batch, train):
    logits = apply_fn({'params': params}, task_split_batch['inputs'])
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=task_split_batch['labels']))


def inner_step(apply_fn, params, task_split_batch, train):
    grads = jax.grad(task_loss_fn, argnums=1)(
        apply_fn, params, task_split_batch, train)

    inner_optimizer = optax.sgd(learning_rate=0.1)
    inner_opt_state = inner_optimizer.init(params)

    updates, inner_opt_state = \
        inner_optimizer.update(grads, inner_opt_state, params)
    params = optax.apply_updates(params, updates)

    return params


def maml_loss_fn(state, params, task_batch, train):
    params_upd = inner_step(
        apply_fn=state.apply_fn,
        params=params,
        task_split_batch=task_batch['train'],
        train=train)

    return task_loss_fn(
        apply_fn=state.apply_fn,
        params=params_upd,
        task_split_batch=task_batch['test'],
        train=train)


def loss_fn(state, params, batch, train):
    maml_loss_fn_vmap = jax.vmap(maml_loss_fn, in_axes=(None, None, 0, None))

    return jnp.mean(maml_loss_fn_vmap(state, params, batch, train))


def task_pred_fn(task_batch, params, model):
    params_upd = inner_step(
        apply_fn=model.apply,
        params=params,
        task_split_batch=task_batch['train'],
        train=False)

    return model.apply(
        {'params': params_upd}, task_batch['test']['inputs']).argmax(axis=-1)


def pred_fn(batch, params, model):
    return jax.vmap(task_pred_fn, in_axes=(0, None, None))(batch, params, model)


def eval_metric_fn(eval_results):
    preds = np.array([result['pred'] for result in eval_results])
    labels = np.array(
        [result['example']['test']['labels'] for result in eval_results])
    return {'acc': np.mean(preds == labels).item()}


def main(dataset_name='omniglot',
         n_ways=5,
         n_shots=5,
         n_test_shots=15,
         n_tasks_per_epoch=1000,
         n_epochs=1000,
         learning_rate=1e-3,
         per_device_batch_size=16,
         jax_seed=42):
    tm_dataset = get_torchmeta_dataset(
        dataset_name=dataset_name,
        n_ways=n_ways,
        n_shots=n_shots,
        n_test_shots=n_test_shots)

    deployer = Deployer(jax_seed=jax_seed)

    model = CNN(n_classes=n_ways)
    dummy_example = sample_tasks(tm_dataset=tm_dataset['train'], n_tasks=1)[0]
    dummy_batch = collate_fn([dummy_example])
    params = model.init(
        deployer.gen_rng(), dummy_batch['train']['inputs'][0])['params']
    optimizer = optax.adam(learning_rate=learning_rate)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lambda x: learning_rate,
        dummy_example=dummy_example)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=partial(pred_fn, model=model),
        output_fn=lambda x: x.tolist(),
        dummy_example=dummy_example)

    eval_examples = sample_tasks(
        tm_dataset=tm_dataset['val'], n_tasks=n_tasks_per_epoch)
    train_examples_fn = partial(
        sample_tasks, tm_dataset=tm_dataset['train'], n_tasks=n_tasks_per_epoch)
    trainer.fit(
        train_examples=train_examples_fn,
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=eval_examples,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn)


if __name__ == '__main__':
    fire.Fire(main)