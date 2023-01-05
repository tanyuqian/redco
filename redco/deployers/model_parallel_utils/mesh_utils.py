import numpy as np

import jax
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from flax.core.frozen_dict import FrozenDict, unfreeze
import optax

from .partition_utils import set_partitions


def get_mesh(mesh_model_shards):
    if mesh_model_shards == 1:
        return None

    assert jax.device_count() % mesh_model_shards == 0

    mesh_devices = np.array(jax.devices()).reshape(
        jax.device_count() // mesh_model_shards, mesh_model_shards)
    mesh = Mesh(mesh_devices, ('dp', 'mp'))

    return mesh


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


def shard_params(params, params_spec, mesh):
    param_init_fn = pjit(
        lambda x: x,
        in_axis_resources=(params_spec, ),
        out_axis_resources=params_spec)

    with mesh:
        params = param_init_fn(params)

    return params


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
