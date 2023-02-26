import numpy as np

import jax
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P
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


class ShapeDtypeStruct:
    __slots__ = ["shape", "dtype"]

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def get_param_spec(params, shard_rules):
    return set_partitions(unfreeze(params), shard_rules)


def shard_params_and_opt_state(params, params_spec, mesh, optimizer):
    def init_fn(params_):
        opt_state_ = optimizer.init(params_)
        return opt_state_, params_

    def get_opt_spec(x):
        if isinstance(x, (dict, FrozenDict,)):
            return params_spec
        return None

    params_shapes = jax.tree_util.tree_map(
        lambda x: ShapeDtypeStruct(x.shape, x.dtype), params)
    state_shapes = jax.eval_shape(init_fn, params_shapes)

    opt_state_spec, _ = jax.tree_util.tree_map(
        get_opt_spec, state_shapes,
        is_leaf=lambda x: isinstance(x, (dict, FrozenDict, optax.EmptyState,)))

    p_get_initial_state = pjit(
        init_fn,
        in_axis_resources=(params_spec,),
        out_axis_resources=(opt_state_spec, params_spec))

    with mesh:
        opt_state, params = p_get_initial_state(params)

    return params, opt_state, opt_state_spec


def gather_params(params, params_spec, mesh):
    gather_fn = pjit(
        lambda x: x, in_axis_resources=(params_spec, ), out_axis_resources=None)

    with mesh:
        return gather_fn(params)


def guess_shard_rules(params, mesh_model_shards, investigate_depth=2):
    shard_rules = {
        ('(bias|scale)',): None,
        ('embedding',): P('mp', None),
    }

    last_dense_mp_dim = None
    flat_params = flatten_dict(params)
    for key in sorted(flat_params.keys(), key=lambda t: (len(t), t)):
        param = flat_params[key]

        rule_key = key[-investigate_depth:]

        if key[-1] in ['bias', 'scale']:
            assert len(param.shape) == 1

        elif key[-1] == 'embedding':
            assert len(param.shape) == 2
            if param.shape[0] % mesh_model_shards != 0:
                shard_rules[('embedding',)] = P(None, 'mp')

        else:
            if len(param.squeeze().shape) == 1:
                shard_rules[rule_key] = None

            elif rule_key in shard_rules:
                for dim_size, rule_str in zip(
                        param.shape, shard_rules[rule_key]):
                    assert rule_str != 'mp' or dim_size % mesh_model_shards == 0

            elif under_attention(key) and rule_key[0][0] == 'o':
                shard_rules[rule_key] = P('mp', None)

            elif under_attention(key) and rule_key[0][0] in ['q', 'k', 'v']:
                shard_rules[rule_key] = P(None, 'mp')

            else:
                rule_tuple = [None for _ in range(len(param.shape))]
                for dim in range(-1, -len(param.shape) - 1, -1):
                    if dim != last_dense_mp_dim and \
                            param.shape[dim] % mesh_model_shards == 0:
                        last_dense_mp_dim = dim
                        rule_tuple[dim] = 'mp'
                        break
                if all([t is None for t in rule_tuple]):
                    if last_dense_mp_dim is not None and \
                            param.shape[last_dense_mp_dim] % \
                            mesh_model_shards == 0:
                        rule_tuple[last_dense_mp_dim] = 'mp'

                shard_rules[rule_key] = P(*rule_tuple)

    return list(shard_rules.items())


def under_attention(flat_param_key):
    for key in flat_param_key:
        if 'attention' in key.lower() or 'attn' in key.lower():
            return True
    return False
