import jax

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_multistep_adamw_optimizer


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

    def get_adamw_optimizer(self,
                            train_size,
                            per_device_batch_size,
                            n_epochs,
                            learning_rate,
                            accumulate_grad_batches,
                            warmup_rate,
                            weight_decay):
        _, global_batch_size = self.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        return get_multistep_adamw_optimizer(
            train_size=train_size,
            global_batch_size=global_batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            accumulate_grad_batches=accumulate_grad_batches,
            warmup_rate=warmup_rate,
            weight_decay=weight_decay)

    @property
    def mesh(self):
        return self._mesh
