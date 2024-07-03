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
#
#  adapted from https://github.com/google-research/google-research/blob/fce923e9dad97cd67492c2a65b9ecdc4b2495204/flax_models/t5x/partitions.py
"""Utilities for constructing PyTrees of PartitionSpecs."""

import re
import numpy as np
import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core.frozen_dict import freeze, unfreeze

# Sentinels
_unmatched = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def set_partitions(in_dict, rules):
    if rules is None:
        return jax.tree.map(lambda _: P(), in_dict)
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))


def get_mesh(n_model_shards):
    if n_model_shards == 1:
        return None

    assert jax.device_count() % n_model_shards == 0

    mesh_devices = np.array(jax.devices()).reshape(
        jax.device_count() // n_model_shards, n_model_shards)
    mesh = Mesh(mesh_devices, ('dp', 'mp'))

    return mesh


def get_params_spec(params_shape_or_params, params_sharding_rules):
    params_spec = jax.tree.map(
        lambda params_sharding_rules_, params_: set_partitions(
            params_, rules=params_sharding_rules_),
        params_sharding_rules, unfreeze(params_shape_or_params),
        is_leaf=lambda x: isinstance(x, list) or x is None)

    return freeze(params_spec)


def shard_params(params, params_spec, mesh):
    if jax.tree.all(jax.tree.map(lambda x: isinstance(
            x, np.ndarray) or x.sharding.is_fully_addressable, params)):
        return jax.tree.map(
            lambda param, param_spec: jax.make_array_from_callback(
                shape=param.shape,
                sharding=jax.sharding.NamedSharding(mesh=mesh, spec=param_spec),
                data_callback=lambda index: param[index]),
            params, params_spec)
    else:
        shard_fn = pjit(lambda x: x, out_shardings=params_spec)
        with mesh:
            return shard_fn(params)


def get_opt_state_spec(params_shape_or_params, params_spec, optimizer):
    def match_params_structure(x):
        try:
            jax.tree.map(lambda x, y: None,  params_spec, x)
        except:
            return False
        return True

    def get_opt_spec(x):
        if match_params_structure(x):
            return jax.tree.map(
                lambda s, xx: s if isinstance(xx, jax.ShapeDtypeStruct) else xx,
                params_spec, x)
        return P()

    return jax.tree.map(
        get_opt_spec, jax.eval_shape(optimizer.init, params_shape_or_params),
        is_leaf=match_params_structure)


def get_sharding_rules(params_shape_or_params, n_model_shards):
    valid_mp_dims, in_attn = {}, {}
    flat_params = flatten_dict(params_shape_or_params)
    for flat_key, param in flat_params.items():
        rule_key = flat_key[-2:] if flat_key[-1] == 'kernel' else flat_key[-1:]

        if flat_key[-1] not in ['embedding', 'kernel']:
            valid_mp_dims[rule_key] = None
            continue
        elif rule_key not in valid_mp_dims:
            valid_mp_dims[rule_key] = [True for _ in param.shape]
            in_attn[rule_key] = True
        elif valid_mp_dims[rule_key] is not None \
                and len(param.shape) != len(valid_mp_dims[rule_key]):
            valid_mp_dims[rule_key] = None
            continue

        for i, dim_size in enumerate(param.shape):
            if dim_size % n_model_shards != 0:
                valid_mp_dims[rule_key][i] = False
        if not any([('attention' in k.lower() or 'attn' in k.lower()) for k in
                    flat_key]):
            in_attn[rule_key] = False

    sharding_rules = {}
    last_mp_dim = None
    for rule_key in sorted(valid_mp_dims.keys()):
        if valid_mp_dims[rule_key] is None or not any(valid_mp_dims[rule_key]):
            sharding_rules[rule_key] = P()
            continue

        valid_idxes = []
        for i, is_valid in enumerate(valid_mp_dims[rule_key]):
            if is_valid:
                valid_idxes.append(i)
        valid_idxes.reverse()

        is_special = False
        if rule_key[-1] == 'embedding':
            is_special = True
            valid_idxes.reverse()
        elif in_attn[rule_key] and (
                rule_key[0][0] in ['q', 'k', 'v']
                or rule_key[0][-1] in ['q', 'k', 'v']):
            is_special = True
        elif in_attn[rule_key] and (
                rule_key[0][0] == 'o' or rule_key[0][-1] == 'o'):
            is_special = True
            valid_idxes.reverse()
        elif (rule_key[0].startswith('up')
              or rule_key[0].startswith('gate')
              or rule_key[0].startswith('wi')
              or 'in' in rule_key[0].split('_')):
            is_special = True
        elif (rule_key[0].startswith('down')
              or rule_key[0].startswith('wo')
              or 'out' in rule_key[0].split('_')):
            is_special = True
            valid_idxes.reverse()
        elif rule_key[0].startswith('head') or rule_key[0].endswith('head'):
            is_special = True

        mp_dim = valid_idxes[0]
        if not is_special and mp_dim == last_mp_dim and len(valid_idxes) > 1:
            mp_dim = valid_idxes[1]
        if not is_special:
            last_mp_dim = mp_dim

        rule_tuple = [None for _ in valid_mp_dims[rule_key]]
        rule_tuple[mp_dim] = 'mp'
        sharding_rules[rule_key] = P(*rule_tuple)

    return list(sharding_rules.items())