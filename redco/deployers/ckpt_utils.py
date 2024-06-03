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
from collections import namedtuple
import json
import os
import re
import gc
import jax
from jax.experimental import multihost_utils
from flax.jax_utils import unreplicate
from flax.core.frozen_dict import freeze, unfreeze
from flax.serialization import (
    msgpack_serialize, msgpack_restore, from_state_dict, to_state_dict)
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node

PARAMS_FILENAME = 'params.msgpack'
PARAMS_INDEX_FILENAME = 'params_index.json'
OPT_STATE_FILENAME = 'opt_state.msgpack'
OPT_STATE_INDEX_FILENAME = 'opt_state_index.json'
RNG_FILENAME = 'rng.msgpack'

RedcoCheckpoint = namedtuple(
    'RedcoCheckpoint', ['params', 'opt_state', 'rng', 'info'])


def load_ckpt(optimizer, ckpt_dir=None, workdir=None):
    if ckpt_dir is None:
        last_ckpt_info_path = f'{workdir}/ckpts/last_ckpt_info.json'
        if not os.path.exists(last_ckpt_info_path):
            return RedcoCheckpoint(None, None, None, None)
        last_ckpt_info = json.load(open(last_ckpt_info_path))
        ckpt_name = last_ckpt_info['last_ckpt_name']
        ckpt_dir = f'{workdir}/ckpts/{ckpt_name}'
    else:
        last_ckpt_info = None

    params = load_params(ckpt_dir=ckpt_dir)
    opt_state = load_opt_state(
        ckpt_dir=ckpt_dir, params=params, optimizer=optimizer)
    rng = load_rng(ckpt_dir=ckpt_dir)

    return RedcoCheckpoint(
        params=params, opt_state=opt_state, rng=rng, info=last_ckpt_info)


def save_params(params,
                ckpt_dir,
                max_shard_size,
                do_unreplicate=False,
                do_process_allgather=False):
    assert not (do_unreplicate and do_process_allgather)
    if do_unreplicate:
        params = unreplicate(params)
        print('UNREPLICATED......')
    if do_process_allgather:
        params = multihost_utils.process_allgather(params)

    if jax.process_index() == 0:
        shards, index = flax_shard_checkpoint(
            params=params,
            filename=PARAMS_FILENAME,
            max_shard_size=max_shard_size)

        if index is None:
            open(f'{ckpt_dir}/{PARAMS_FILENAME}', "wb").write(
                msgpack_serialize(unfreeze(params)))
        else:
            json.dump(index, open(
                f'{ckpt_dir}/{PARAMS_INDEX_FILENAME}', 'w'), indent=4)
            for shard_file, shard in shards.items():
                open(f'{ckpt_dir}/{shard_file}', "wb").write(msgpack_serialize(
                    unfreeze(unflatten_dict(shard, sep='/'))))


def save_opt_state(opt_state,
                   ckpt_dir,
                   max_shard_size,
                   do_unreplicate=False,
                   do_process_allgather=False):
    assert not (do_unreplicate and do_process_allgather)
    if do_unreplicate:
        opt_state = unreplicate(opt_state)
    if do_process_allgather:
        opt_state = multihost_utils.process_allgather(opt_state)

    if jax.process_index() == 0:
        opt_state = to_state_dict(opt_state)
        shards, index = flax_shard_checkpoint(
            params=opt_state,
            filename=OPT_STATE_FILENAME,
            max_shard_size=max_shard_size)

        if index is None:
            open(f'{ckpt_dir}/{OPT_STATE_FILENAME}', "wb").write(
                msgpack_serialize(unfreeze(opt_state)))
        else:
            json.dump(index, open(
                f'{ckpt_dir}/{OPT_STATE_INDEX_FILENAME}', 'w'), indent=4)
            for shard_file, shard in shards.items():
                open(f'{ckpt_dir}/{shard_file}', "wb").write(
                    msgpack_serialize(unfreeze(unflatten_dict(shard, sep='/'))))


def save_rng(rng, ckpt_dir):
    if jax.process_index() == 0:
        open(f'{ckpt_dir}/{RNG_FILENAME}', "wb").write(msgpack_serialize(rng))


def load_params(ckpt_dir):
    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        if os.path.exists(f'{ckpt_dir}/{PARAMS_INDEX_FILENAME}'):
            weight_map = json.load(
                open(f'{ckpt_dir}/{PARAMS_INDEX_FILENAME}'))['weight_map']
            shard_files = [
                f'{ckpt_dir}/{filename}'
                for filename in sorted(list(set(weight_map.values())))
            ]
            params = load_flax_sharded_weights(shard_files=shard_files)
        else:
            params = msgpack_restore(
                open(f'{ckpt_dir}/{PARAMS_FILENAME}', 'rb').read())

    return params


def load_opt_state(ckpt_dir, params, optimizer):
    if (not os.path.exists(f'{ckpt_dir}/{OPT_STATE_FILENAME}') and
            not os.path.exists(f'{ckpt_dir}/{OPT_STATE_INDEX_FILENAME}')):
        return None

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        if os.path.exists(f'{ckpt_dir}/{OPT_STATE_INDEX_FILENAME}'):
            weight_map = json.load(open(
                f'{ckpt_dir}/{OPT_STATE_INDEX_FILENAME}'))['weight_map']
            shard_files = [
                f'{ckpt_dir}/{filename}'
                for filename in sorted(list(set(weight_map.values())))
            ]
            opt_state = load_flax_sharded_weights(shard_files=shard_files)
        else:
            opt_state = msgpack_restore(
                open(f'{ckpt_dir}/{OPT_STATE_FILENAME}', 'rb').read())

        params_shapes = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), freeze(params))

        return from_state_dict(
            target=jax.eval_shape(optimizer.init, params_shapes),
            state=opt_state)


def load_rng(ckpt_dir):
    return msgpack_restore(open(f'{ckpt_dir}/{RNG_FILENAME}', 'rb').read())


def load_flax_sharded_weights(shard_files):
    """
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flax_utils.py#L460

    This is the same as [`flax.serialization.from_bytes`]
    (https:lax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        shard_files (`List[str]`:
            The list of shard files to load.

    Returns:
        `Dict`: A nested dictionary of the model parameters, in the expected format for flax models : `{'model':
        {'params': {'...'}}}`.
    """

    # Load the index
    state_sharded_dict = {}
    for shard_file in shard_files:
        # load using msgpack utils
        state = msgpack_restore(open(shard_file, 'rb').read())
        state = flatten_dict(state, keep_empty_nodes=True, sep="/")
        state_sharded_dict.update(state)
        del state
        gc.collect()

    # the state dict is unflattened to the match the format of model.params
    return unflatten_dict(state_sharded_dict, sep="/")


def convert_file_size_to_int(size):
    """
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L914

    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2 ** 30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2 ** 20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2 ** 10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10 ** 9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10 ** 6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10 ** 3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError(
        "`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def dtype_byte_size(dtype):
    """
    Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flax_utils.py#L84

    Returns the size (in bytes) occupied by one parameter of type `dtype`. Example:
    ```py
    >>> dtype_byte_size(np.float32)
    4
    ```
    """
    if dtype == bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", dtype.name)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def flax_shard_checkpoint(params, filename, max_shard_size="10GB"):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size. The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so
    there is no optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For
    example, if the limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as
    [6GB], [6+2GB], [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        params (`Union[Dict, FrozenDict]`): A `PyTree` of model parameters.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    # flatten the weights to chunk
    weights = flatten_dict(params, keep_empty_nodes=True, sep="/")
    for item in weights:
        if weights[item] is empty_node:
            current_block[item] = empty_node
            continue

        weight_size = weights[item].size * dtype_byte_size(weights[item].dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        current_block[item] = weights[item]
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {filename: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = filename.replace(
            ".msgpack",
            f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.msgpack")
        shards[shard_file] = shard
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index
