import copy
import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard


def get_batch(examples, preprocess_fn):
    processed_examples = [preprocess_fn(example) for example in examples]

    model_inputs = {}
    for key in processed_examples[0].keys():
        elements = [
            np.expand_dims(example[key], axis=0)
            for example in processed_examples]
        model_inputs[key] = np.concatenate(elements, axis=0)

    return model_inputs


def get_dataloader(examples, batch_size, preprocess_fn, do_shard):
    for i in range(0, len(examples) // batch_size):
        batch = get_batch(
            examples=examples[i * batch_size:(i + 1) * batch_size],
            preprocess_fn=preprocess_fn)
        yield {
            key: shard(jnp.array(value)) if do_shard else jnp.array(value)
            for key, value in batch.items()
        }


def get_data_batches(examples, batch_size, preprocess_fn, do_shard, desc):
    data_loader = get_dataloader(
        examples=examples,
        batch_size=batch_size,
        preprocess_fn=preprocess_fn,
        do_shard=do_shard)
    return tqdm.tqdm(
        data_loader, total=len(examples) // batch_size, desc=desc)


def shuffle_examples(examples, shuffle_rng):
    shuffled_idxes = jax.random.permutation(key=shuffle_rng, x=len(examples))
    return [examples[idx] for idx in shuffled_idxes]


def get_host_examples(examples, global_batch_size, shuffle, shuffle_rng, mesh):
    if shuffle:
        examples = shuffle_examples(examples=examples, shuffle_rng=shuffle_rng)

    examples = examples[:len(examples) // global_batch_size * global_batch_size]

    for i in range(len(examples)):
        examples[i] = copy.deepcopy(examples[i])
        examples[i].update({'example_id': i})

    if mesh is None:
        return examples[jax.process_index()::jax.process_count()]
    else:
        raise NotImplementedError
