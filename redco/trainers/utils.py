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
from jax.example_libraries.optimizers import l2_norm


def train_step(train_rng,
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

    if mesh is None:
        loss, grads = loss_and_grads(batch)
        grads = jax.lax.pmean(grads, axis_name='dp')
    else:
        loss, grads = jax.vmap(loss_and_grads)(batch)
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


def eval_step(state, batch, loss_fn, mesh, compute_dtype):
    def get_loss(batch_):
        return loss_fn(
            train_rng=jax.random.PRNGKey(0),
            state=state,
            params=jax.tree.map(
                lambda x: x.astype(compute_dtype), state.params),
            batch=batch_,
            is_training=False)

    if mesh is None:
        loss = jax.lax.pmean(get_loss(batch), axis_name='dp')
    else:
        loss = jnp.mean(jax.vmap(get_loss)(batch))

    return {'loss': loss}