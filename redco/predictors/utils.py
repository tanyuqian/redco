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
from jax.experimental.multihost_utils import process_allgather


def add_idxes(examples):
    return [
        {'example': example, 'idx': idx} for idx, example in enumerate(examples)
    ]


def collate_fn_wrapper(examples, collate_fn):
    return {
        'batch': collate_fn([example['example'] for example in examples]),
        'idxes': np.array([example['idx'] for example in examples])
    }


def pred_fn_wrapper(rng, params, batch, pred_fn):
    return {
        'preds': pred_fn(rng=rng, params=params, batch=batch['batch']),
        'idxes': batch['idxes']
    }


def pred_step(rng, params, batch, pred_fn, mesh):
    if mesh is None:
        return pred_fn(rng=rng, params=params, batch=batch)
    else:
        return jax.vmap(
            lambda rng_, batch_: pred_fn(rng=rng_, params=params, batch=batch_)
        )(jax.random.split(rng, num=mesh.shape['dp']), batch)


def process_batch_preds(batch_preds_with_idxes, mesh):
    if mesh is None and jax.process_count() > 1:
        batch_preds_with_idxes = jax.tree.map(
            lambda t: t.reshape((-1,) + t.shape[2:]),
            process_allgather(batch_preds_with_idxes))

    batch_preds_with_idxes = jax.tree.map(np.asarray, batch_preds_with_idxes)
    preds = jax.tree.map(
        lambda t: t.reshape((-1,) + t.shape[2:]),
        batch_preds_with_idxes['preds'])
    idxes_argsort = np.argsort(batch_preds_with_idxes['idxes'].reshape(-1))

    return jax.tree.map(lambda t: t[idxes_argsort], preds)


def default_output_fn(preds):
    batch_size = jax.tree.leaves(preds)[0].shape[0]
    assert jax.tree.all(jax.tree.map(lambda x: x.shape[0] == batch_size, preds))
    return [jax.tree.map(lambda x: x[i], preds) for i in range(batch_size)]
