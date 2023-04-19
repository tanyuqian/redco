#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import jax
import optax


def collate_fn(examples, train_key='train', val_key='test'):
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


def loss_fn(train_rng,
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


def pred_fn(pred_rng,
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
