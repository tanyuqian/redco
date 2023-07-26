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
from optax._src.linear_algebra import global_norm


def default_loss_and_grads(train_rng, state, batch, loss_fn):
    def compute_loss(params):
        return loss_fn(
            train_rng=train_rng,
            state=state,
            params=params,
            batch=batch,
            is_training=True)

    grad_fn = jax.value_and_grad(compute_loss)
    return grad_fn(state.params)


def default_train_step(train_rng,
                       state,
                       batch,
                       loss_fn,
                       lr_schedule_fn,
                       under_pmap):
    loss, grads = default_loss_and_grads(
        train_rng=train_rng, state=state, batch=batch, loss_fn=loss_fn)

    if under_pmap:
        grads = jax.lax.pmean(grads, 'batch')

    new_state = state.apply_gradients(grads=grads)

    metrics = {
        'loss': loss,
        'step': state.step,
        'grad_norm': global_norm(grads)
    }
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
