import copy
import tqdm
import multiprocessing

import numpy as np
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard


def collage_batch(collate_fn, examples):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    processed_examples = pool.map(
        collate_fn, [[example] for example in examples])

    batch = {}
    for key in processed_examples[0].keys():
        elements = [example[key] for example in processed_examples]
        batch[key] = np.concatenate(elements, dim=0)

    return batch


def get_dataloader(examples, batch_size, collate_fn, do_shard):
    for i in range(0, len(examples) // batch_size):
        batch=collate_batch(
            collate_fn=collate_fn,
            examples=examples[i * batch_size:(i + 1) * batch_size])
        yield {
            key: shard(jnp.array(value)) if do_shard else jnp.array(value)
            for key, value in batch.items()
        }


def get_data_batches(examples, batch_size, collate_fn, do_shard, desc):
    data_loader = get_dataloader(
        examples=examples,
        batch_size=batch_size,
        collate_fn=collate_fn,
        do_shard=do_shard)
    return tqdm.tqdm(
        data_loader, total=len(examples) // batch_size, desc=desc)


def shuffle_examples(examples, shuffle_rng):
    shuffled_idxes = jax.random.permutation(key=shuffle_rng, x=len(examples))
    return [examples[int(idx)] for idx in shuffled_idxes]


def get_host_examples(examples, global_batch_size, shuffle, shuffle_rng, mesh):
    if shuffle:
        examples = shuffle_examples(examples=examples, shuffle_rng=shuffle_rng)

    examples = examples[:len(examples) // global_batch_size * global_batch_size]

    if mesh is None:
        return examples[jax.process_index()::jax.process_count()]
    else:
        raise NotImplementedError
