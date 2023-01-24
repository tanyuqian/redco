import jax
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard_prng_key

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_multistep_adamw_optimizer
from .log_utils import get_logger

from .model_parallel_utils.mesh_utils import (
    get_mesh,
    get_host_batch_size,
    shard_params_and_opt_state,
    get_param_spec,
    guess_shard_rules)


class Deployer:
    def __init__(self, jax_seed, mesh_model_shards=1, verbose=True):
        self._rng = jax.random.PRNGKey(seed=jax_seed)
        self._verbose = verbose
        self._mesh = get_mesh(mesh_model_shards=mesh_model_shards)
        self._logger = get_logger(verbose=verbose)

    def process_batch_size(self, per_device_batch_size):
        if self._mesh is None:
            batch_size = per_device_batch_size * jax.local_device_count()
            global_batch_size = batch_size * jax.process_count()
        else:
            global_batch_size = per_device_batch_size * self._mesh.shape['dp']
            batch_size = get_host_batch_size(
                global_batch_size=global_batch_size, mesh=self._mesh)

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
            desc=f'{desc} (global_batch_size = {global_batch_size})',
            verbose=self._verbose)

    def process_batch_preds(self, batch_preds):
        if self._mesh is None:
            return jax.tree_util.tree_map(
                lambda t: t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]),
                batch_preds)
        else:
            return batch_preds

    def process_to_run_model(self, x, is_prng_key=False):
        if self._mesh is None:
            if is_prng_key:
                return shard_prng_key(x)
            else:
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

    def guess_shard_rules(self, params):
        if self._mesh is None:
            return None
        else:
            shard_rules = guess_shard_rules(
                params=params, mesh_model_shards=self._mesh.shape['mp'])

            self.log(
                info='\n'.join([f'{t}' for t in shard_rules]),
                title='Guessed shard rules')

            return shard_rules

    def get_params_spec(self, params, shard_rules):
        return get_param_spec(params=params, shard_rules=shard_rules)

    def shard_params_and_opt_state(self, params, params_spec, optimizer):
        return shard_params_and_opt_state(
            params=params,
            params_spec=params_spec,
            mesh=self._mesh,
            optimizer=optimizer)

    def run_model_step(self, step_fn, input_args):
        if self._mesh is None:
            return step_fn(*input_args)
        else:
            with self._mesh:
                return step_fn(*input_args)

    def gen_rng(self):
        self._rng, new_rng = jax.random.split(self._rng)
        return new_rng

    def log(self, info, title=None):
        if title is not None:
            boarder_len

            self._logger.info(f'=== {title}')
            for t in info.split('\n'):
                self._logger.info(t)
            self._logger.info('=' * 64)
        else:
            self._logger.info(info)

    @property
    def mesh(self):
        return self._mesh
