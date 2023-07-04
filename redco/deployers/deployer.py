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

import os
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard_prng_key
from flax.core.frozen_dict import unfreeze
from flax.serialization import msgpack_serialize, msgpack_restore

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_lr_schedule_fn
from .log_utils import get_logger, log_info, save_outputs
from .model_parallel_utils.mesh_utils import (
    get_mesh,
    shard_params_and_opt_state,
    shard_params,
    gather_params_to_cpu,
    get_param_spec,
    get_sharding_rules)


class Deployer:
    def __init__(self,
                 jax_seed,
                 n_model_shards=1,
                 verbose=True,
                 workdir=None,
                 run_tensorboard=False):
        if workdir is not None:
            os.makedirs(workdir, exist_ok=True)

        self._verbose = verbose
        self._workdir = workdir
        self._logger = get_logger(verbose=verbose, workdir=workdir)

        if run_tensorboard and jax.process_index() == 0:
            from flax.metrics import tensorboard
            self._summary_writer = tensorboard.SummaryWriter(workdir)
        else:
            self._summary_writer = None

        self._rng = jax.random.PRNGKey(seed=jax_seed)
        self._mesh = get_mesh(n_model_shards=n_model_shards)

    def process_batch_size(self, per_device_batch_size):
        if self._mesh is None:
            batch_size = per_device_batch_size * jax.local_device_count()
            global_batch_size = batch_size * jax.process_count()
        else:
            batch_size = per_device_batch_size * self._mesh.shape['dp']
            global_batch_size = batch_size

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

    def process_batch_preds(self, batch_preds_with_idxes):
        if self._mesh is None:
            batch_preds_with_idxes = jax.tree_util.tree_map(
                lambda x: x[0], batch_preds_with_idxes)

            batch_preds = batch_preds_with_idxes['raw_preds']
            idxes = batch_preds_with_idxes['__idx__']

            preds = jax.tree_util.tree_map(
                lambda t: t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]),
                batch_preds)
            idxes = idxes.reshape(-1)
            idxes_argsort = jnp.argsort(idxes, axis=None)

            return jax.tree_util.tree_map(lambda t: t[idxes_argsort], preds)
        else:
            return batch_preds_with_idxes['raw_preds']

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

    def get_lr_schedule_fn(self,
                           train_size,
                           per_device_batch_size,
                           n_epochs,
                           learning_rate,
                           schedule_type='linear',
                           warmup_rate=0.,
                           warmup_steps=None,
                           init_learning_rate=0.,
                           end_learning_rate=0.):
        _, global_batch_size = self.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        total_train_steps = n_epochs * (train_size // global_batch_size)

        if warmup_steps is None:
            warmup_steps = int(total_train_steps * warmup_rate)

        return get_lr_schedule_fn(
            schedule_type=schedule_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            init_learning_rate=init_learning_rate,
            learning_rate=learning_rate,
            end_learning_rate=end_learning_rate)

    def get_sharding_rules(self, params):
        if self._mesh is None:
            return None
        else:
            sharding_rules = get_sharding_rules(
                params=params, mesh_model_shards=self._mesh.shape['mp'])

            self.log_info(
                info='\n'.join([f'{t}' for t in sharding_rules]),
                title='Sharding rules')

            return sharding_rules

    def get_params_spec(self, params, params_sharding_rules):
        return get_param_spec(
            params=params, params_sharding_rules=params_sharding_rules)

    def shard_params(self, params, params_spec):
        return shard_params(
            params=params, params_spec=params_spec, mesh=self._mesh)

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

    def log_info(self, info, title=None, step=None):
        if jax.process_index() == 0:
            log_info(
                info=info,
                title=title,
                logger=self._logger,
                summary_writer=self._summary_writer,
                step=step)

    def log_metrics(self, metrics, step):
        if self._summary_writer is not None:
            for metric_name, value in metrics.items():
                self._summary_writer.scalar(metric_name, value, step=step)

    def save_outputs(self, outputs, desc, step):
        if self._workdir is not None and jax.process_index() == 0:
            save_outputs(
                workdir=self._workdir,
                outputs=outputs,
                desc=desc,
                step=step,
                logger=self._logger,
                summary_writer=self._summary_writer)

    def load_params(self, filepath):
        params = msgpack_restore(open(filepath, 'rb').read())
        self.log_info(f'params loaded from {filepath}')

        return params

    def save_params(self, params, filepath, params_sharding_rules=None):
        if self._mesh is not None:
            params_spec = self.get_params_spec(
                params=params, params_sharding_rules=params_sharding_rules)
            params = gather_params_to_cpu(
                params=params, params_spec=params_spec, mesh=self._mesh)

        if jax.process_index() == 0:
            save_dir = '/'.join(filepath.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)

            open(filepath, "wb").write(msgpack_serialize(unfreeze(params)))
            self.log_info(f'params saved into {filepath}')

    @property
    def mesh(self):
        return self._mesh

    @property
    def workdir(self):
        return self._workdir
