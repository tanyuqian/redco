import jax

from .data_utils import get_host_examples, get_data_batches


class Deployer:
    def __init__(self, workdir):
        self._n_model_shards = 1
        self._mesh = None

        self._workdir = workdir

    def process_batch_size(self, per_device_batch_size):
        assert self._n_model_shards == 1
        batch_size = per_device_batch_size * jax.local_device_count()
        global_batch_size = batch_size * jax.process_count()

        return batch_size, global_batch_size

    def get_host_examples(self,
                          examples,
                          global_batch_size,
                          shuffle, shuffle_rng):
        return get_host_examples(
            examples=examples,
            global_batch_size=global_batch_size,
            shuffle=shuffle,
            shuffle_rng=shuffle_rng,
            mesh=self._mesh)

    @staticmethod
    def get_data_batches(examples, batch_size, preprocess_fn, desc):
        return get_data_batches(
            examples=examples,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn,
            desc=desc)

    @property
    def mesh(self):
        return self._mesh
