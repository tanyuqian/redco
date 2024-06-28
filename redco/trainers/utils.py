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

import jax
import jax.numpy as jnp


def global_norm(updates):
    return jnp.sqrt(sum(
        jnp.sum(x * x) for x in jax.tree_util.tree_leaves(updates)
    ))


def loss_and_grads(train_rng, state, batch, loss_fn):
    def compute_loss(params):
        return loss_fn(
            train_rng=train_rng,
            state=state,
            params=params,
            batch=batch,
            is_training=True)

    grad_fn = jax.value_and_grad(compute_loss)
    return grad_fn(state.params)


def train_step(train_rng, state, batch, loss_fn, lr_schedule_fn, mesh):
    if mesh is None:
        loss, grads = loss_and_grads(
            train_rng=train_rng, state=state, batch=batch, loss_fn=loss_fn)
        grads = jax.lax.pmean(grads, 'batch')
    else:
        loss, grads = jax.vmap(lambda b: loss_and_grads(
            train_rng=train_rng, state=state, batch=b, loss_fn=loss_fn))(batch)
        loss = jnp.mean(loss)
        grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)

    new_state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': loss, 'step': state.step, 'grad_norm': global_norm(grads)
    }
    if lr_schedule_fn is not None:
        metrics.update({'lr': lr_schedule_fn(state.step)})

    if mesh is None:
        metrics = jax.lax.pmean(metrics, axis_name='batch')

    return new_state, metrics


def eval_step(state, batch, loss_fn, mesh):
    if mesh is None:
        loss = loss_fn(train_rng=jax.random.PRNGKey(0),
                state=state,
                params=state.params,
                batch=batch,
                is_training=False)
        loss = jax.lax.pmean(loss, axis_name='batch')
    else:
        loss = jax.vmap(
            lambda b: loss_fn(
                train_rng=jax.random.PRNGKey(0),
                state=state,
                params=state.params,
                batch=b,
                is_training=False))(batch)
        loss = jnp.mean(loss)

    return {'loss': loss}