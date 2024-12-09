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
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.example_libraries.optimizers import l2_norm


def default_train_step(
        rng, state, batch, loss_fn, lr_schedule_fn, mesh, compute_dtype):
    def loss_and_grads(rng_, batch_):
        if mesh is not None:
            rng_ = jnp.squeeze(rng_, axis=0)
            batch_ = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), batch_)

        loss, grads = jax.value_and_grad(
            lambda params: loss_fn(
                rng=rng_,
                state=state,
                params=params,
                batch=batch_,
                is_training=True)
        )(jax.tree.map(lambda x: x.astype(compute_dtype), state.params))

        if mesh is not None:
            loss, grads = loss[None], jax.tree.map(lambda x: x[None], grads)

        return loss, grads

    if mesh is None:
        loss, grads = loss_and_grads(rng, batch)
        grads = jax.lax.pmean(grads, axis_name='dp')
    else:
        loss_and_grads_fn = shard_map(
            loss_and_grads,
            mesh=mesh,
            in_specs=(P('dp'), P('dp')),
            out_specs=(P('dp'), P('dp'))
        )
        loss, grads = loss_and_grads_fn(
            jax.random.split(rng, num=mesh.shape['dp']), batch)
        loss = jnp.mean(loss, axis=0)
        grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)

    new_state = state.apply_gradients(grads=jax.tree.map(
        lambda grad, param: grad.astype(param.dtype), grads, state.params))

    metrics = {'loss': loss, 'step': state.step, 'grad_norm': l2_norm(grads)}
    if lr_schedule_fn is not None:
        metrics['lr'] = lr_schedule_fn(state.step)
    if mesh is None:
        metrics = jax.lax.pmean(metrics, axis_name='dp')

    return new_state, metrics


def eval_step(rng, state, batch, loss_fn, mesh, compute_dtype):
    def get_loss(rng_, batch_):
        return loss_fn(
            rng=rng_,
            state=state,
            params=jax.tree.map(
                lambda x: x.astype(compute_dtype), state.params),
            batch=batch_,
            is_training=False)

    if mesh is None:
        loss = jax.lax.pmean(get_loss(rng, batch), axis_name='dp')
    else:
        loss = jnp.mean(jax.vmap(get_loss)(
            jax.random.split(rng, num=mesh.shape['dp']), batch))

    return {'loss': loss}