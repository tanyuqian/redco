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
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.traverse_util import flatten_dict
import optax

from .partition_utils import set_partitions


def get_mesh(n_model_shards):
    if n_model_shards == 1:
        return None

    assert jax.device_count() % n_model_shards == 0

    mesh_devices = np.array(jax.devices()).reshape(
        jax.device_count() // n_model_shards, n_model_shards)
    mesh = Mesh(mesh_devices, ('dp', 'mp'))

    return mesh


def get_param_spec(params, params_sharding_rules):
    return set_partitions(unfreeze(params), params_sharding_rules)


def shard_params(params, params_spec, mesh):
    shard_fn = pjit(
        lambda x: x, in_shardings=(params_spec,), out_shardings=params_spec)

    with mesh:
        return shard_fn(params)


def get_opt_state_spec(params, params_spec, optimizer):
    def init_fn(params_):
        return optimizer.init(params_)

    def get_opt_spec(x):
        if isinstance(x, (dict, FrozenDict,)):
            return params_spec
        return None

    params_shapes = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)

    return jax.tree_util.tree_map(
        get_opt_spec, jax.eval_shape(init_fn, params_shapes),
        is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,)))


def get_sharding_rules(params, n_model_shards):
    def get_valid_mp_dims(param):
        result = []
        for i, dim in enumerate(param.shape):
            if dim % n_model_shards == 0:
                result.append(i)
        result.reverse()
        return result

    def inside_attention(flat_key):
        for k in flat_key:
            if 'attention' in k.lower() or 'attn' in k.lower():
                return True
        return False

    sharding_rules = {}
    flat_params = flatten_dict(params)
    last_shard_dim = None
    for flat_key in sorted(flat_params.keys(), key=lambda t: (len(t), t[-2:])):
        rule_key, param = flat_key[-2:], flat_params[flat_key]

        if rule_key[-1] not in ['kernel', 'embedding']:
            sharding_rules[flat_key[-1:]] = None
            continue

        if rule_key[-1] == 'embedding':
            rule_key = rule_key[-1:]

        if rule_key in sharding_rules:
            if sharding_rules[rule_key] is None:
                continue
            else:
                for dim_size, rule_str in zip(
                        param.shape, sharding_rules[rule_key]):
                    assert rule_str != 'mp' or dim_size % n_model_shards == 0
        elif len(param.shape) != 2:
            sharding_rules[rule_key] = None
        else:
            valid_mp_dims = get_valid_mp_dims(param)
            is_special = False

            if rule_key[-1] == 'embedding':
                is_special = True
            elif inside_attention(flat_key) and (
                    rule_key[0][0] in ['q', 'k', 'v'] or \
                    rule_key[0][-1] in ['q', 'k', 'v']):
                is_special = True
            elif inside_attention(flat_key) and (
                    rule_key[0][0] == 'o' or rule_key[0][-1] == 'o'):
                is_special = True
                valid_mp_dims.reverse()
            elif flat_key[-2].startswith('up') or \
                    flat_key[-2].startswith('gate') or \
                    flat_key[-2].startswith('wi'):
                is_special = True
            elif flat_key[-2].startswith('down') or \
                    flat_key[-2].startswith('wo'):
                is_special = True
                valid_mp_dims.reverse()
            elif flat_key[-2].startswith('head') or \
                    flat_key[-2].endswith('head'):
                is_special = True

            if len(valid_mp_dims) > 0:
                shard_dim = valid_mp_dims[0]
                if is_special == False and \
                        shard_dim == last_shard_dim and len(valid_mp_dims) > 1:
                    shard_dim = valid_mp_dims[1]
            else:
                shard_dim = None

            rule_tuple = [None] * len(param.shape)
            if shard_dim is not None:
                rule_tuple[shard_dim] = 'mp'
            sharding_rules[rule_key] = P(*rule_tuple)

            if not is_special:
                last_shard_dim = shard_dim

    return list(sharding_rules.items())
