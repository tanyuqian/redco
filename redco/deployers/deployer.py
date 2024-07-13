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
import orbax.checkpoint as ocp

from .data_utils import get_host_examples, get_data_batches
from .opt_utils import get_lr_schedule_fn
from .log_utils import get_logger, log_info, save_outputs
from .ckpt_utils import save_ckpt, load_params_shape, load_ckpt
from .partition_utils import (
    get_mesh,
    get_params_spec,
    get_sharding_rules,
    get_opt_state_spec,
    shard_params)


DEFAULT_HOST0_PORT = 11111


class Deployer:
    def __init__(self,
                 jax_seed,
                 n_model_shards=1,
                 verbose=True,
                 workdir=None,
                 n_processes=None,
                 host0_address=None,
                 host0_port=None,
                 process_id=None,
                 n_local_devices=None,
                 run_tensorboard=False,
                 wandb_init_kwargs=None):
        if n_processes is None:
            if 'SLURM_JOB_NUM_NODES' in os.environ:
                n_processes = int(os.environ['SLURM_JOB_NUM_NODES'])
                process_id = int(os.environ['SLURM_NODEID'])
            else:
                n_processes = 1

        if n_processes > 1:
            local_device_ids = None if n_local_devices is None \
                else list(range(n_local_devices))

            if host0_port is None:
                host0_port = DEFAULT_HOST0_PORT

            jax.distributed.initialize(
                coordinator_address=f'{host0_address}:{host0_port}',
                num_processes=n_processes,
                process_id=process_id,
                local_device_ids=local_device_ids)

        if workdir is not None:
            os.makedirs(workdir, exist_ok=True)

        self._verbose = verbose
        self._workdir = workdir
        self._logger = get_logger(verbose=verbose, workdir=workdir)

        if wandb_init_kwargs is not None and jax.process_index() == 0:
            import wandb
            wandb.init(**wandb_init_kwargs)
            self._wandb_log_fn = wandb.log
        else:
            self._wandb_log_fn = None

        if run_tensorboard and jax.process_index() == 0:
            from flax.metrics import tensorboard
            self._summary_writer = tensorboard.SummaryWriter(workdir)
        else:
            self._summary_writer = None

        self.log_info(
            f'Local Devices: {jax.local_device_count()} / {jax.device_count()}')

        self._rng = jax.random.PRNGKey(seed=jax_seed)
        self._mesh = get_mesh(n_model_shards=n_model_shards)
        self._checkpointer = ocp.PyTreeCheckpointer()

    def get_local_global_micro_batch_size(self, per_device_batch_size):
        if self._mesh is None:
            local_micro_batch_size = \
                per_device_batch_size * jax.local_device_count()
            global_micro_batch_size = \
                local_micro_batch_size * jax.process_count()
        else:
            global_micro_batch_size = local_micro_batch_size = \
                per_device_batch_size * self._mesh.shape['dp']

        return local_micro_batch_size, global_micro_batch_size

    def get_accumulate_grad_batches(
            self, global_batch_size, per_device_batch_size):
        _, global_micro_batch_size = self.get_local_global_micro_batch_size(
            per_device_batch_size=per_device_batch_size)
        assert global_batch_size % global_micro_batch_size == 0
        accumulate_grad_batches = global_batch_size // global_micro_batch_size

        return accumulate_grad_batches

    def get_model_input_batches(self,
                                examples,
                                per_device_batch_size,
                                collate_fn,
                                shuffle,
                                shuffle_rng,
                                desc,
                                is_train=False,
                                accumulate_grad_batches=None):
        local_micro_batch_size, global_micro_batch_size = \
            self.get_local_global_micro_batch_size(
                per_device_batch_size=per_device_batch_size)

        examples = get_host_examples(
            examples=examples,
            global_micro_batch_size=global_micro_batch_size,
            shuffle=shuffle,
            shuffle_rng=shuffle_rng,
            mesh=self._mesh)

        if not is_train:
            desc = f'{desc} (global_batch_size = {global_micro_batch_size})'
        elif accumulate_grad_batches is None:
            desc = \
                f'{desc} (global_micro_batch_size = {global_micro_batch_size})'
        else:
            desc = (f'{desc} ('
                    f'global_micro_batch_size = {global_micro_batch_size}, '
                    f'accumulate_grad_batches = {accumulate_grad_batches})')

        return get_data_batches(
            examples=examples,
            batch_size=local_micro_batch_size,
            collate_fn=collate_fn,
            mesh=self._mesh,
            desc=desc,
            verbose=self._verbose)

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
        _, global_micro_batch_size = self.get_local_global_micro_batch_size(
            per_device_batch_size=per_device_batch_size)
        total_train_steps = n_epochs * (train_size // global_micro_batch_size)

        if warmup_steps is None:
            warmup_steps = int(total_train_steps * warmup_rate)

        return get_lr_schedule_fn(
            schedule_type=schedule_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            init_learning_rate=init_learning_rate,
            learning_rate=learning_rate,
            end_learning_rate=end_learning_rate)

    def get_sharding_rules(self, params_shape_or_params):
        if self._mesh is None:
            return None
        else:
            sharding_rules = get_sharding_rules(
                params_shape_or_params=params_shape_or_params,
                n_model_shards=self._mesh.shape['mp'])
            return sharding_rules

    def get_params_spec(self, params_shape_or_params, params_sharding_rules):
        return get_params_spec(
            params_shape_or_params=params_shape_or_params,
            params_sharding_rules=params_sharding_rules)

    def get_opt_state_spec(self,
                           params_shape_or_params,
                           params_spec,
                           optimizer):
        return get_opt_state_spec(
            params_shape_or_params=params_shape_or_params,
            params_spec=params_spec,
            optimizer=optimizer)

    def shard_params(self, params, params_spec, desc='params'):
        self.log_info(info=f'Sharding {desc} ...')
        return shard_params(
            mesh=self._mesh, params=params, params_spec=params_spec)

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

        if self._wandb_log_fn is not None:
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

    def save_ckpt(self,
                  ckpt_dir,
                  params,
                  opt_state=None,
                  float_dtype=None,
                  **kwargs):
        ckpt_dir = os.path.abspath(ckpt_dir)
        self.log_info(f'Saving ckpt to {ckpt_dir} ...')
        save_ckpt(
            ckpt_dir=ckpt_dir,
            checkpointer=self._checkpointer,
            params=params,
            opt_state=opt_state,
            float_dtype=float_dtype,
            rng=self._rng,
            **kwargs)
        self.log_info(f'Ckpt saved into {ckpt_dir}')

    def load_params_shape(self, ckpt_dir):
        return load_params_shape(ckpt_dir=ckpt_dir)

    def load_ckpt(self,
                  ckpt_dir,
                  params_sharding_rules=None,
                  optimizer=None,
                  float_dtype=None,
                  load_params=True,
                  load_opt_state=True,
                  update_rng=False):
        ckpt_dir = os.path.abspath(ckpt_dir)
        self.log_info(f'Loading ckpt from {ckpt_dir} ...')

        params_shape = self.load_params_shape(ckpt_dir=ckpt_dir)

        specs = {}
        if self._mesh is not None:
            if params_sharding_rules is None:
                params_sharding_rules = self.get_sharding_rules(
                    params_shape_or_params=params_shape)

            specs['params'] = self.get_params_spec(
                params_shape_or_params=params_shape,
                params_sharding_rules=params_sharding_rules)
            if optimizer is not None:
                specs['opt_state'] = self.get_opt_state_spec(
                    params_shape_or_params=params_shape,
                    params_spec=specs['params'],
                    optimizer=optimizer)

        ckpt, info = load_ckpt(
            ckpt_dir=ckpt_dir,
            checkpointer=self._checkpointer,
            params_shape_or_params=params_shape,
            optimizer=optimizer,
            float_dtype=float_dtype,
            mesh=self._mesh,
            specs=specs,
            load_params=load_params,
            load_opt_state=load_opt_state)

        for key, value in info.items():
            if not update_rng and key == 'rng':
                continue
            self.log_info(f'{ckpt_dir}::{key} = {value}')

        if update_rng:
            self._rng = info['rng']
            self.log_info(f'rng updated to {self._rng} (by {ckpt_dir})')

        return ckpt, info

    def load_last_ckpt(self,
                       optimizer=None,
                       params_sharding_rules=None,
                       float_dtype=None,
                       load_params=True,
                       load_opt_state=True,
                       update_rng=True):
        try:
            last_ckpt_name = open(
                f'{self._workdir}/ckpts/last_ckpt.txt').read().strip()
        except:
            self.log_info(
                f'{self._workdir}/ckpts/last_ckpt.txt not found. '
                f'no ckpt loaded.')
            return None, None

        return self.load_ckpt(
            ckpt_dir=f'{self._workdir}/ckpts/{last_ckpt_name}',
            optimizer=optimizer,
            float_dtype=float_dtype,
            params_sharding_rules=params_sharding_rules,
            load_params=load_params,
            load_opt_state=load_opt_state,
            update_rng=update_rng)

    @property
    def mesh(self):
        return self._mesh

    @property
    def workdir(self):
        return self._workdir
