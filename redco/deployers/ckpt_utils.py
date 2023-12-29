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
import os
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from flax.jax_utils import unreplicate
from flax.core.frozen_dict import freeze, unfreeze
from flax.serialization import (
    msgpack_serialize, msgpack_restore, from_state_dict, to_state_dict)


def save_params(mesh, params, ckpt_dir):
    filepath = f'{ckpt_dir}/params.msgpack'
    if mesh is not None:
        params = multihost_utils.process_allgather(params)
    if jax.process_index() == 0:
        open(filepath, "wb").write(msgpack_serialize(unfreeze(params)))


def save_opt_state(mesh, opt_state, ckpt_dir):
    filepath = f'{ckpt_dir}/opt_state.msgpack'

    if mesh is None:
        opt_state = unreplicate(opt_state)
    else:
        opt_state = multihost_utils.process_allgather(opt_state)

    if jax.process_index() == 0:
        opt_state = to_state_dict(opt_state)
        open(filepath, "wb").write(msgpack_serialize(unfreeze(opt_state)))


def save_rng(rng, ckpt_dir):
    if jax.process_index() == 0:
        jnp.save(f'{ckpt_dir}/rng.npy', rng)


def load_params(ckpt_dir):
    filepath = f'{ckpt_dir}/params.msgpack'
    with jax.default_device(jax.devices('cpu')[0]):
        params = msgpack_restore(open(filepath, 'rb').read())
        params = jax.tree_util.tree_map(jnp.asarray, params)

    return params


def load_opt_state(ckpt_dir, params, optimizer):
    filepath = f'{ckpt_dir}/opt_state.msgpack'
    if not os.path.exists(filepath):
        return None

    with jax.default_device(jax.devices('cpu')[0]):
        opt_state = msgpack_restore(open(filepath, 'rb').read())
        opt_state = from_state_dict(
            target=optimizer.init(freeze(params)), state=opt_state)
    return opt_state


def load_rng(ckpt_dir):
    return jnp.load(f'{ckpt_dir}/rng.npy')
