from functools import partial
import fire
import jax.numpy as jnp
import optax
import numpy as np

from redco import Deployer, MAMLTrainer, MAMLPredictor

from utils import CNN, get_torchmeta_dataset, sample_tasks


def inner_loss_fn(params, batch, model):
    logits = model.apply({'params': params}, batch['inputs'])
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['labels']))


def inner_pred_fn(batch, params, model):
    return model.apply({'params': params}, batch['inputs']).argmax(axis=-1)


def eval_metric_fn(eval_results):
    preds = np.array([result['pred'] for result in eval_results])
    labels = np.array(
        [result['example']['test']['labels'] for result in eval_results])
    return {'acc': np.mean(preds == labels).item()}


def main(dataset_name='omniglot',
         n_ways=5,
         n_shots=5,
         n_test_shots=15,
         n_tasks_per_epoch=10000,
         n_epochs=1000,
         learning_rate=1e-3,
         per_device_batch_size=16,
         inner_learning_rate=0.1,
         inner_n_steps=1,
         train_key='train',
         val_key='test',
         jax_seed=42):
    tm_dataset = get_torchmeta_dataset(
        dataset_name=dataset_name,
        n_ways=n_ways,
        n_shots=n_shots,
        n_test_shots=n_test_shots)

    deployer = Deployer(jax_seed=jax_seed)

    model = CNN(n_classes=n_ways)
    dummy_example = sample_tasks(tm_dataset=tm_dataset['train'], n_tasks=1)[0]
    params = model.init(deployer.gen_rng(), np.array(
        dummy_example['train']['inputs']))['params']
    optimizer = optax.adam(learning_rate=learning_rate)

    trainer = MAMLTrainer(
        deployer=deployer,
        apply_fn=model.apply,
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lambda t: learning_rate,
        inner_loss_fn=partial(inner_loss_fn, model=model),
        inner_learning_rate=inner_learning_rate,
        inner_n_steps=inner_n_steps,
        dummy_example=dummy_example,
        train_key=train_key,
        val_key=val_key)

    predictor = MAMLPredictor(
        deployer=deployer,
        inner_loss_fn=partial(inner_loss_fn, model=model),
        inner_learning_rate=inner_learning_rate,
        inner_n_steps=inner_n_steps,
        inner_pred_fn=partial(inner_pred_fn, model=model),
        output_fn=lambda x: x.tolist(),
        dummy_example=dummy_example,
        train_key=train_key, val_key=val_key)

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