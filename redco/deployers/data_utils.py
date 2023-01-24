import tqdm

import jax
import jax.numpy as jnp
from flax.training.common_utils import shard


def get_dataloader(examples, batch_size, collate_fn, do_shard):
    def make_jnp(value):
        return jax.tree_util.tree_map(jnp.asarray, value)

    for i in range(0, len(examples) // batch_size):
        batch = collate_fn(
            examples=examples[i * batch_size:(i + 1) * batch_size])
        yield {
            key: shard(make_jnp(value)) if do_shard else make_jnp(value)
            for key, value in batch.items()
        }


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
