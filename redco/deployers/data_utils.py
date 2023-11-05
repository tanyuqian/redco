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

import tqdm
from collections import UserDict
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard


def get_dataloader(examples, batch_size, collate_fn, do_shard):
    def make_jnp(value):
        value = jax.tree_util.tree_map(
            lambda x: dict(x) if isinstance(x, UserDict) else x, value)
        value = jax.tree_util.tree_map(jnp.asarray, value)
        if do_shard:
            value = jax.tree_util.tree_map(shard, value)
        return value

    for i in range(0, len(examples) // batch_size):
        yield make_jnp(collate_fn(
            examples=examples[i * batch_size:(i + 1) * batch_size]))


def get_data_batches(examples,
                     batch_size,
                     collate_fn,
                     do_shard,
                     desc,
                     verbose):
    data_loader = get_dataloader(
        examples=examples,
        batch_size=batch_size,
        collate_fn=collate_fn,
        do_shard=do_shard)
    return tqdm.tqdm(
        data_loader,
        total=len(examples) // batch_size,
        desc=desc,
        disable=(not verbose))


def get_host_examples(examples, global_batch_size, shuffle, shuffle_rng, mesh):
    if shuffle:
        shuffled_idxes = jax.random.permutation(
            key=shuffle_rng, x=len(examples))
        examples = [examples[int(idx)] for idx in shuffled_idxes]

    examples = examples[:len(examples) // global_batch_size * global_batch_size]

    if mesh is None:
        return examples[jax.process_index()::jax.process_count()]
    else:
        return examples