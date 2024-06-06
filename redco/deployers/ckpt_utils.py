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
import json
from flax.core.frozen_dict import freeze
from flax.training import orbax_utils
import orbax.checkpoint as ocp


def save_ckpt(checkpointer,
              ckpt_dir,
              params,
              opt_state=None,
              rng=None,
              **kwargs):
    ckpt = {'params': params, 'opt_state': opt_state}
    for key in ['params', 'opt_state']:
        if ckpt[key] is not None:
            checkpointer.save(
                f'{ckpt_dir}/{key}',
                ckpt[key],
                save_args=orbax_utils.save_args_from_target(ckpt[key]),
                force=True)

    if jax.process_index() == 0:
        if rng is not None:
            kwargs['rng'] = rng.tolist()
        json.dump(kwargs, open(f'{ckpt_dir}/info.json', 'w'), indent=4)


def load_ckpt(checkpointer,
              ckpt_dir,
              optimizer=None,
              params_shape=None,
              mesh=None,
              specs=None,
              load_params=True,
              load_opt_state=True):
    keys_to_load = []
    if load_params and os.path.exists(f'{ckpt_dir}/params'):
        keys_to_load.append('params')
    if load_opt_state and os.path.exists(f'{ckpt_dir}/opt_state'):
        keys_to_load.append('opt_state')

    ckpt = {'params': None, 'opt_state': None}
    for key in keys_to_load:
        print(f'Restoring {ckpt_dir}/{key} ..')

        if key == 'opt_state':
            assert optimizer is not None, \
                (f'optimizer and params_shape must not be None '
                 f'because ckpt {ckpt_dir} has opt_state')
            target_shape = jax.eval_shape(optimizer.init, params_shape)
        else:
            target_shape = params_shape

        if mesh is None:
            assert params_shape is not None, \
                'params_shape must not be None when mesh is None'
            restore_args = jax.tree_util.tree_map(
                lambda param: ocp.ArrayRestoreArgs(
                    sharding=jax.sharding.SingleDeviceSharding(
                        jax.local_devices()[0])
                ), target_shape
            )
        else:
            assert specs is not None and key in specs, \
                f'specs[{key}] must not be None when mesh is not None'
            restore_args = jax.tree_util.tree_map(
                lambda spec: ocp.ArrayRestoreArgs(
                    sharding=jax.sharding.NamedSharding(mesh=mesh, spec=spec)
                ), specs[key]
            )

        ckpt[key] = checkpointer.restore(
            f'{ckpt_dir}/{key}',
            args=ocp.args.PyTreeRestore(
                item=target_shape, restore_args=restore_args)
        )

    info = json.load(open(f'{ckpt_dir}/info.json'))
    if 'rng' in info:
        info['rng'] = jnp.array(info['rng'], dtype=jnp.uint32)

    return ckpt, info
