import copy
import numpy as np
import jax


def add_idxes(examples):
    examples_upd = []
    for idx, example in enumerate(examples):
        assert '__idx__' not in example
        example = copy.deepcopy(example)
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