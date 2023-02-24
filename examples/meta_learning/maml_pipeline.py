import numpy as np
import jax
import optax


def maml_collate_fn(examples, train_key='train', val_key='test'):
    return {
        'train': {
            key: np.stack([example[train_key][key] for example in examples])
            for key in examples[0][train_key].keys()
        },
        'val': {
            key: np.stack([example[val_key][key] for example in examples])
            for key in examples[0][val_key].keys()
        }
    }


def inner_step(params,
               inner_batch,
               inner_loss_fn,
               inner_learning_rate,
               inner_n_steps):
    grads = jax.grad(inner_loss_fn)(params, inner_batch)

    inner_optimizer = optax.sgd(learning_rate=inner_learning_rate)
    inner_opt_state = inner_optimizer.init(params)

    for _ in range(inner_n_steps):
        updates, inner_opt_state = inner_optimizer.update(
            updates=grads, state=inner_opt_state, params=params)
        params = optax.apply_updates(params, updates)

    return params


def maml_loss_fn(train_rng,
                 state,
                 params,
                 batch,
                 is_training,
                 inner_loss_fn,
                 inner_learning_rate,
                 inner_n_steps):
    def inner_maml_loss_fn(inner_batch_train, inner_batch_val):
        params_upd = inner_step(
            params=params,
            inner_batch=inner_batch_train,
            inner_loss_fn=inner_loss_fn,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps)

        return inner_loss_fn(params=params_upd, batch=inner_batch_val)

    return jax.vmap(inner_maml_loss_fn)(batch['train'], batch['val']).mean()


def maml_pred_fn(pred_rng,
                 batch,
                 params,
                 inner_loss_fn,
                 inner_learning_rate,
                 inner_n_steps,
                 inner_pred_fn):
    def inner_maml_pred_fn(inner_batch_train, inner_batch_val):
        params_upd = inner_step(
            params=params,
            inner_batch=inner_batch_train,
            inner_loss_fn=inner_loss_fn,
            inner_learning_rate=inner_learning_rate,
            inner_n_steps=inner_n_steps)

        return inner_pred_fn(batch=inner_batch_val, params=params_upd)

    return jax.vmap(inner_maml_pred_fn)(batch['train'], batch['val'])
