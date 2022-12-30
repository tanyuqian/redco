import jax
from flax.jax_utils import replicate, unreplicate

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_multistep_adamw_optimizer


class Deployer:
    def __init__(self, jax_seed):
        self._n_model_shards = 1
        self._mesh = None

        self._rng = jax.random.PRNGKey(seed=jax_seed)

    def process_batch_size(self, per_device_batch_size):
        assert self._n_model_shards == 1
        batch_size = per_device_batch_size * jax.local_device_count()
        global_batch_size = batch_size * jax.process_count()

        return batch_size, global_batch_size

    def get_model_input_batches(self,
                                examples,
                                per_device_batch_size,
                                collate_fn,
                                shuffle,
                                shuffle_rng,
                                desc):
        batch_size, global_batch_size = self.process_batch_size(
            per_device_batch_size=per_device_batch_size)

        examples = get_host_examples(
            examples=examples,
            global_batch_size=global_batch_size,
            shuffle=shuffle,
            shuffle_rng=shuffle_rng,
            mesh=self._mesh)

        return get_data_batches(
            examples=examples,
            batch_size=batch_size,
            collate_fn=collate_fn,
            do_shard=(self.mesh is None),
            desc=desc)

    def process_batch_preds(self, batch_preds):
        if self._mesh is None:
            return jax.tree_util.tree_map(
                lambda t: t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]),
                batch_preds)
        else:
            return batch_preds

    def process_to_run_model(self, x):
        if self._mesh is None:
            return replicate(x)
        else:
            return x

    def process_to_deliver(self, x):
        if self._mesh is None:
            return unreplicate(x)
        else:
            return x

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

    def gen_rng(self):
        self._rng, new_rng = jax.random.split(self._rng)
        return new_rng

    @property
    def mesh(self):
        return self._mesh
