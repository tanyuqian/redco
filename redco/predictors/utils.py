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
from flax.jax_utils import unreplicate


def add_idxes(examples):
    assert all(['__idx__' not in example for example in examples])
    return [{**example, '__idx__': idx} for idx, example in enumerate(examples)]


def collate_fn_wrapper(examples, collate_fn):
    idxes = [example.pop('__idx__') for example in examples]
    return {**collate_fn(examples), '__idx__': np.array(idxes)}


def pred_fn_wrapper(pred_rng, params, batch, pred_fn):
    idxes = batch.pop('__idx__')
    return {
        'raw_preds': pred_fn(pred_rng=pred_rng, params=params, batch=batch),
        '__idx__': idxes
    }


def pred_step(pred_rng, params, batch, pred_fn, mesh):
    if mesh is None:
        preds = pred_fn(pred_rng=pred_rng, params=params, batch=batch)
        return jax.lax.all_gather(preds, axis_name='batch')
    else:
        return jax.vmap(lambda b: pred_fn(
            pred_rng=pred_rng, params=params, batch=b))(batch)


def default_output_fn(preds):
    batch_size = jax.tree.leaves(preds)[0].shape[0]
    assert jax.tree.all(jax.tree.map(lambda x: x.shape[0] == batch_size, preds))
    return [jax.tree.map(lambda x: x[i], preds) for i in range(batch_size)]


def process_batch_preds(batch_preds_with_idxes, mesh):
    if mesh is None:
        batch_preds_with_idxes = unreplicate(batch_preds_with_idxes)

    batch_preds_with_idxes = jax.tree.map(np.asarray, batch_preds_with_idxes)
    preds = jax.tree.map(
        lambda t: t.reshape((-1,) + t.shape[2:]),
        batch_preds_with_idxes['raw_preds'])
    idxes_argsort = np.argsort(batch_preds_with_idxes['__idx__'].reshape(-1))

    return jax.tree.map(lambda t: t[idxes_argsort], preds)