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
from flax.core.frozen_dict import freeze
from flax.training import orbax_utils


def save_ckpt(checkpointer, ckpt_dir, params, opt_state, rng):
    ckpt = {'params': params, 'opt_state': opt_state, 'rng': {'rng': rng}}
    for key, value in ckpt.items():
        print(f'KEY: {key}')
        if value is not None:
            checkpointer.save(
                f'{ckpt_dir}/{key}',
                value,
                save_args=orbax_utils.save_args_from_target(value),
                force=True)


def load_ckpt(checkpointer, ckpt_dir, optimizer):
    ckpt = {}
    for key in ['params', 'opt_state', 'rng']:
        print(f'LOADING: {key}')
        if os.path.exists(f'{ckpt_dir}/{key}'):
            if key == 'opt_state':
                params_shapes = jax.tree_util.tree_map(
                    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                    freeze(ckpt['params']))
                kwargs = {'item': jax.eval_shape(optimizer.init, params_shapes)}
            else:
                kwargs = {}

            ckpt[key] = checkpointer.restore(f'{ckpt_dir}/{key}', **kwargs)

    ckpt['rng'] = ckpt['rng']['rng']

    return ckpt
