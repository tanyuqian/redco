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
from jax.experimental import multihost_utils
from flax.jax_utils import replicate, unreplicate
from flax.training.common_utils import shard_prng_key
from flax.core.frozen_dict import freeze, unfreeze
from flax.serialization import (
    msgpack_serialize, msgpack_restore, from_state_dict, to_state_dict)

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_lr_schedule_fn
from .log_utils import get_logger, log_info, save_outputs
from .model_parallel_utils.mesh_utils import (
    get_mesh,
    shard_params_and_opt_state,
    shard_params,
    get_param_spec,
    get_sharding_rules)


class Deployer:
    def __init__(self,
                 jax_seed,
                 n_model_shards=1,
                 verbose=True,
                 workdir=None,
                 n_processes=1,
                 host0_address=None,
                 host0_port=None,
                 process_id=None,
                 n_local_devices=None,
                 run_tensorboard=False,
                 run_wandb=False):
        if n_processes > 1:
            local_device_ids = None if n_local_devices is None \
                else list(range(n_local_devices))

            jax.distributed.initialize(
                coordinator_address=f'{host0_address}:{host0_port}',
                num_processes=n_processes,
                process_id=process_id,
                local_device_ids=local_device_ids)

            print(f'process_id: {jax.process_index()} / {jax.process_count()}')
            print(f'devices: {jax.local_device_count()} / {jax.device_count()}')

        if workdir is not None:
            os.makedirs(workdir, exist_ok=True)

        self._verbose = verbose
        self._workdir = workdir
        self._logger = get_logger(verbose=verbose, workdir=workdir)

        if run_wandb:
            import wandb
            self._wandb_log_fn = wandb.log
        else:
            self._wandb_log_fn = None

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
                                desc,
                                is_train=False,
                                accumulate_grad_batches=None):
        batch_size, global_batch_size = self.process_batch_size(
            per_device_batch_size=per_device_batch_size)

        examples = get_host_examples(
            examples=examples,
            global_batch_size=global_batch_size,
            shuffle=shuffle,
            shuffle_rng=shuffle_rng,
            mesh=self._mesh)

        if not is_train:
            desc = f'{desc} (global_batch_size = {global_batch_size})'
        elif accumulate_grad_batches is None:
            desc = f'{desc} (global_micro_batch_size = {global_batch_size})'
        else:
            desc = (f'{desc} ('
                    f'global_micro_batch_size = {global_batch_size}, '
                    f'accumulate_grad_batches = {accumulate_grad_batches})')

        return get_data_batches(
            examples=examples,
            batch_size=batch_size,
            collate_fn=collate_fn,
            do_shard=(self.mesh is None),
            desc=desc,
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

    def shard_params_and_opt_state(
            self, params, params_spec, optimizer, init_opt_state):
        return shard_params_and_opt_state(
            params=params,
            params_spec=params_spec,
            mesh=self._mesh,
            optimizer=optimizer,
            init_opt_state=init_opt_state)

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

        if self._wandb_log_fn is not None and jax.process_index() == 0:
            self._wandb_log_fn(metrics, step)

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
        with jax.default_device(jax.devices('cpu')[0]):
            params = msgpack_restore(open(filepath, 'rb').read())
            params = jax.tree_util.tree_map(jnp.asarray, params)

        self.log_info(f'params loaded from {filepath}')
        return params

    def load_opt_state(self, ckpt_dir, desc, params, optimizer):
        filepath = f'{ckpt_dir}/opt_state_{desc}.msgpack'
        if not os.path.exists(filepath):
            self.log_info(f'Skip loading opt_state (No file {filepath})')
            return None

        with jax.default_device(jax.devices('cpu')[0]):
            opt_state = msgpack_restore(open(filepath, 'rb').read())
            opt_state = from_state_dict(
                target=optimizer.init(freeze(params)), state=opt_state)

        self.log_info(f'opt_state loaded from {filepath}.')
        return opt_state

    def save_params(self, params, ckpt_dir, desc):
        if self._mesh is None:
            params = unreplicate(params)
        else:
            with jax.default_device(jax.devices('cpu')[0]):
                params = multihost_utils.process_allgather(params)

        if jax.process_index() == 0:
            filepath = f'{ckpt_dir}/params_{desc}.msgpack'
            open(filepath, "wb").write(msgpack_serialize(unfreeze(params)))
            self.log_info(f'params saved into {filepath}')

    def save_opt_state(self, opt_state, ckpt_dir, desc):
        if self._mesh is None:
            opt_state = unreplicate(opt_state)
        else:
            with jax.default_device(jax.devices('cpu')[0]):
                opt_state = multihost_utils.process_allgather(opt_state)

        if jax.process_index() == 0:
            filepath = f'{ckpt_dir}/opt_state_{desc}.msgpack'
            opt_state = to_state_dict(opt_state)
            open(filepath, "wb").write(msgpack_serialize(unfreeze(opt_state)))
            self.log_info(f'opt_state saved into {filepath}')

    def save_rng(self, ckpt_dir, desc):
        if jax.process_index() == 0:
            jnp.save(f'{ckpt_dir}/rng_{desc}.npy', self._rng)

    @property
    def mesh(self):
        return self._mesh

    @property
    def workdir(self):
        return self._workdir
