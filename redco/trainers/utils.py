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
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale


class TrainState(train_state.TrainState):
    dynamic_scale: DynamicScale


def default_loss_and_grads(train_rng, state, batch, loss_fn, under_pmap):
    def compute_loss(params):
        return loss_fn(
            train_rng=train_rng,
            state=state,
            params=params,
            batch=batch,
            is_training=True)

    grad_fn = state.dynamic_scale.value_and_grad(
        compute_loss, axis_name='batch' if under_pmap else None)

    return grad_fn(state.params)


def default_train_step(train_rng,
                       state,
                       batch,
                       loss_fn,
                       lr_schedule_fn,
                       params_grad_weights,
                       under_pmap):
    dynamic_scale, is_finite, loss, grads = default_loss_and_grads(
        train_rng=train_rng,
        state=state,
        batch=batch,
        loss_fn=loss_fn,
        under_pmap=under_pmap)

    grads = jax.tree_util.tree_map(lambda x: jnp.where(is_finite, x, 0.), grads)

    if params_grad_weights is not None:
        grads = jax.tree_util.tree_map(
            lambda x, y: x * y, grads, params_grad_weights)

    # if under_pmap:
    #     grads = jax.lax.pmean(grads, 'batch')

    new_state = state.apply_gradients(grads=grads, dynamic_scale=dynamic_scale)

    metrics = {'loss': loss, 'step': state.step, 'grad_is_finite': is_finite}
    if lr_schedule_fn is not None:
        metrics.update({'lr': lr_schedule_fn(state.step)})

    if under_pmap:
        metrics = jax.lax.pmean(metrics, axis_name='batch')

    return new_state, metrics


def default_eval_step(state, batch, loss_fn, under_pmap):
    loss = loss_fn(
        train_rng=jax.random.PRNGKey(0),
        state=state,
        params=state.params,
        batch=batch,
        is_training=False)

    metrics = {'loss': loss}
    if under_pmap:
        metrics = jax.lax.pmean(metrics, axis_name='batch')

    return metrics
