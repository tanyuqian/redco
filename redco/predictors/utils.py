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

import copy
import numpy as np
import jax


def add_idxes(examples):
    examples_upd = []
    for idx, example in enumerate(examples):
        assert '__idx__' not in example
        # example = copy.deepcopy(example)
        example = {key: example[key] for key in example.keys()}
        example.update({'__idx__': idx})

        examples_upd.append(example)

    return examples_upd


def collate_fn_wrapper(examples, collate_fn):
    idxes = [example.pop('__idx__') for example in examples]
    batch = collate_fn(examples)
    batch['__idx__'] = np.array(idxes)

    return batch


def pred_fn_wrapper(pred_rng, params, batch, pred_fn, under_pmap):
    idxes = batch.pop('__idx__')
    preds = pred_fn(pred_rng=pred_rng, params=params, batch=batch)
    preds = {
        'raw_preds': preds,
        '__idx__': idxes
    }

    if under_pmap:
        return jax.lax.all_gather(preds, axis_name='batch')
    else:
        return preds


def default_output_fn(preds):
    batch_size = jax.tree_util.tree_leaves(preds)[0].shape[0]

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: x.shape[0] == batch_size, preds))

    outputs = []
    for i in range(batch_size):
        outputs.append(jax.tree_util.tree_map(lambda x: x[i], preds))

    return outputs