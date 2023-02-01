import numpy as np

import jax
import jax.numpy as jnp
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


def get_mesh_process_matrix(mesh):
    return np.asarray(jax.tree_map(
        lambda x: x.process_index, mesh.devices.tolist()))


def get_process_mesh_idx(mesh, process_idx):
    process_matrix = get_mesh_process_matrix(mesh)
    idxes_dp, idxes_mp = np.where(process_matrix == process_idx)

    return [min(idxes_dp) // len(set(idxes_dp)),
            min(idxes_mp) // len(set(idxes_mp))]


def get_host_batch_size(global_batch_size, mesh):
    assert not mesh.is_multi_process
    return global_batch_size


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

    params = get_host_params(params=params, params_spec=params_spec, mesh=mesh)

    with mesh:
        opt_state, params = p_get_initial_state(params)

    return params, opt_state, opt_state_spec


def get_host_params(params, params_spec, mesh):
    p_model_init_fn = pjit(
        lambda: params,
        in_axis_resources=(),
        out_axis_resources=params_spec)

    with mesh:
        host_param_shapes = jax.eval_shape(p_model_init_fn)

    param_shard_idx = \
        get_process_mesh_idx(mesh=mesh, process_idx=jax.process_index())[1]

    def split_param(host_param_shape, param):
        param_shape = np.array(param.shape)
        host_param_shape = np.array(host_param_shape)
        dim_mask = (param_shape != host_param_shape).astype(int)

        return jax.lax.dynamic_slice(
            param,
            start_indices=dim_mask * host_param_shape * param_shard_idx,
            slice_sizes=host_param_shape)

    with jax.default_device(jax.devices('cpu')[0]):
        return jax.tree_util.tree_map(split_param, host_param_shapes, params)


def under_attention(flat_param_key):
    return any([('attention' in t.lower()
                 or 'attn' in t.lower()) for t in flat_param_key])


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
