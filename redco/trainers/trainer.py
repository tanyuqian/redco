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
from functools import partial
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from flax.core.frozen_dict import freeze
from orbax.checkpoint.utils import \
    fully_replicated_host_local_array_to_global_array

from .utils import default_train_step, eval_step


class Trainer:
    """Trainer class managing distributed training process.

    Attributes:
        step (int): Current training step.
        workdir (str): Working directory for saving checkpoints and logs.
        mesh (jax Mesh): Mesh used for distributed training.
        state (flax TrainState): Current training state.
    """
    def __init__(self,
                 deployer,
                 collate_fn,
                 apply_fn,
                 loss_fn,
                 params,
                 optimizer,
                 opt_state=None,
                 compute_dtype=jnp.float32,
                 last_ckpt_info=None,
                 lr_schedule_fn=None,
                 accumulate_grad_batches=None,
                 params_sharding_rules=None,
                 train_step_fn=None):
        """Initializes the Trainer with initial parameters, etc.

        Args:
            deployer (Deployer): A deployer supporting low-level operations.
            collate_fn (Callable): The function converting a data batch to model
                inputs, e.g., tokenizing sentences into input_ids.
            apply_fn (Callable): The function to apply the model, such as
                model.apply for Flax modules, or model itself for HuggingFace
                models. It would be set as state.apply_fn, and used in loss_fn.
            loss_fn (Callable): The loss function converting model inputs to a
                scalar loss, e.g., computing cross-entropy loss from input_ids.
            params (dict): Initial model parameters.
            optimizer (optax optimizer): The optimizer used for training.
            opt_state (dict): optimizer state.
            compute_dtype (dtype): Computation dtype, e.g., `jnp.bfloat16`,
                independent of param dtypes. (for mixed-precision training)
            last_ckpt_info (dict): the beginning step and epoch.
            lr_schedule_fn (Callable): The learning rate schedule
                function converting step to learning rate.
            accumulate_grad_batches (int): Gradient accumulation step.
            params_sharding_rules (list): Sharding rules.
            train_step_fn (Callable): For fully customizing every training step,
                e.g., per-sample gradient noising for data-private training.
        """
        self._deployer = deployer
        self._collate_fn = collate_fn
        self._apply_fn = apply_fn
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._compute_dtype = compute_dtype
        self._lr_schedule_fn = lr_schedule_fn
        self._accumulate_grad_batches = accumulate_grad_batches
        self._params_sharding_rules = params_sharding_rules
        self._train_step_fn = train_step_fn

        self._state = None
        self._state_spec = None
        self._p_train_step = None
        self._p_eval_step = None

        self._init_step = 0
        self._init_epoch_idx = 0
        if last_ckpt_info is not None:
            self._init_step = last_ckpt_info.get('step', 0)
            self._init_epoch_idx = last_ckpt_info.get('epoch_idx', -1) + 1

        n_params = sum([param.size for param in jax.tree.leaves(params)])
        self._deployer.log_info(f'{n_params:,}', title='Parameters')

        self.set_train_state(
            apply_fn=self._apply_fn,
            params=params,
            optimizer=self._optimizer,
            step=self._init_step,
            opt_state=opt_state)

    def set_train_state(
            self, apply_fn, params, optimizer, step, opt_state=None):
        """Sets/Resets the training state with given parameters and optimizer.

        Args:
            apply_fn (Callable): The function to apply the model.
            params (dict): Model parameters.
            optimizer (dict): The optimizer used for training.
            step (int): The training step.
            opt_state (dict): The state of the optimizer.
        """
        self._deployer.log_info('Setting train_state ...')
        params = freeze(params)

        if self.mesh is None:
            params = jax.device_put(params, jax.local_devices()[0])
            if opt_state is None:
                self._deployer.log_info('Initializing opt_state ...')
                opt_state = optimizer.init(params)
            else:
                opt_state = jax.device_put(opt_state, jax.local_devices()[0])

            self._state = train_state.TrainState(
                step=step,
                apply_fn=apply_fn,
                params=params,
                tx=optimizer,
                opt_state=opt_state)
            self._state = replicate(self._state)
        else:
            params_spec = self._deployer.get_params_spec(
                params_shape_or_params=params,
                params_sharding_rules=self._params_sharding_rules)
            params = self._deployer.shard_params(
                params=params, params_spec=params_spec)

            if opt_state is None:
                self._deployer.log_info('Initializing opt_state ...')
                opt_state = optimizer.init(params)

            opt_state_spec = self._deployer.get_opt_state_spec(
                params_shape_or_params=params,
                params_spec=params_spec,
                optimizer=optimizer)
            opt_state = self._deployer.shard_params(
                params=opt_state,
                params_spec=opt_state_spec,
                desc='opt_state')

            self._state = train_state.TrainState(
                apply_fn=apply_fn,
                params=params,
                tx=optimizer,
                opt_state=opt_state,
                step=step)

            self._state_spec = train_state.TrainState(
                apply_fn=apply_fn,
                params=params_spec,
                tx=optimizer,
                opt_state=opt_state_spec,
                step=None)

    def setup_running_step(self, dummy_batch):
        """Sets up the running step functions for training and evaluation.

        Args:
            dummy_batch (PyTree): A dummy batch of data.
        """
        train_step_fn = partial(
            self._train_step_fn or default_train_step,
            loss_fn=self._loss_fn,
            lr_schedule_fn=self._lr_schedule_fn,
            mesh=self.mesh,
            compute_dtype=self._compute_dtype)
        eval_step_fn = partial(
            eval_step,
            loss_fn=self._loss_fn,
            mesh=self.mesh,
            compute_dtype=self._compute_dtype)

        if self.mesh is None:
            self._p_train_step = jax.pmap(train_step_fn, axis_name='dp')
            self._p_eval_step = jax.pmap(eval_step_fn, axis_name='dp')
        else:
            data_spec = jax.tree.map(lambda x: P('dp'), dummy_batch)
            self._p_train_step = pjit(
                train_step_fn,
                in_shardings=(None, self._state_spec, data_spec),
                out_shardings=(self._state_spec, None),
                donate_argnums=(1, ))
            self._p_eval_step = pjit(
                eval_step_fn,
                in_shardings=(self._state_spec, data_spec),
                out_shardings=None)

    def train(self, examples, per_device_batch_size, desc=None):
        """Trains the model on the provided examples.

        Args:
            examples (list): Training examples in python list.
            per_device_batch_size (int): The batch size per device.
            desc (str): Description in the progress bar.
        """
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            shuffle_rng=self._deployer.gen_rng(),
            desc=f'Training ({desc})' if desc is not None else 'Training',
            is_train=True,
            accumulate_grad_batches=self._accumulate_grad_batches)

        for batch in data_batches:
            if self._p_train_step is None:
                self.setup_running_step(dummy_batch=batch)

            train_rng = self._deployer.gen_rng()
            if self.mesh is None:
                train_rng = jax.random.split(
                    train_rng, num=jax.process_count())[jax.process_index()]
                train_rng = shard_prng_key(train_rng)
            self._state, metrics = self._deployer.run_model_step(
                step_fn=self._p_train_step,
                input_args=(train_rng, self._state, batch))

            if self.mesh is None:
                metrics = unreplicate(metrics)
            data_batches.set_postfix(**metrics)
            self._deployer.log_metrics(metrics=metrics, step=self.step)

    def eval_loss(self, examples, per_device_batch_size, desc=None):
        """Evaluates the loss on the provided examples.

        Args:
            examples (list): Evaluation examples in list.
            per_device_batch_size (int): The batch size per device.
            desc (str): Description in the progress bar.

        Returns:
            (float): The average loss over the evaluation examples.
        """
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            shuffle_rng=None,
            desc=f'Evaluating ({desc})' if desc is not None else 'Evaluating')

        losses = []
        for batch in data_batches:
            if self._p_eval_step is None:
                self.setup_running_step(dummy_batch=batch)

            metrics = self._deployer.run_model_step(
                step_fn=self._p_eval_step, input_args=(self._state, batch))

            if self.mesh is None:
                metrics = unreplicate(metrics)

            losses.append(metrics['loss'].item())
            data_batches.set_postfix(**metrics)

        return np.mean(losses).item()

    def fit(self,
            train_examples,
            per_device_batch_size,
            n_epochs,
            eval_examples=None,
            eval_per_device_batch_size=None,
            eval_loss=True,
            eval_predictor=None,
            eval_metric_fn=None,
            eval_sanity_check=True,
            save_every_ckpt=False,
            save_last_ckpt=False,
            save_argmin_ckpt_by_metrics=None,
            save_argmax_ckpt_by_metrics=None,
            save_opt_states=True,
            save_float_dtype=None):
        """Fits the model on the training data for a given number of epochs,
        optionally evaluating and saving checkpoints.

        Args:
            train_examples (list or Callable): Training examples, can be a
                list or a function of epoch_idx (for assigning different
                examples in separate epochs/chunks),
                e.g., `train_examples=lambda epoch_idx: load_data(chunk_idx)`
            per_device_batch_size (int): The batch size per device.
            n_epochs (int): Number of epochs to train.
            eval_examples (list): Examples for evaluation and prediction.
            eval_per_device_batch_size (int): Batch size for evaluation
            eval_loss (bool): Whether to evaluate loss.
            eval_predictor (Predictor): Predictor working on `eval_examples`.
            eval_metric_fn (Callable): Metric function for prediction.
            eval_sanity_check (bool): if to run a sanity check for
                evaluation & predict functions before training.
            save_every_ckpt (bool): if to save a ckpt after every epoch.
            save_last_ckpt (bool): Whether to save the last checkpoint.
            save_argmin_ckpt_by_metrics (list[str]): Metrics to save checkpoints
                based on minimum values.
            save_argmax_ckpt_by_metrics (list[str]): Metrics to save checkpoints
                based on maximum values.
            save_opt_states (bool): of to save optimizer states in ckpts.
            save_float_dtype (bool): The data type for saving checkpoints.
        """
        if eval_per_device_batch_size is None:
            eval_per_device_batch_size = per_device_batch_size

        if save_argmax_ckpt_by_metrics is None:
            save_argmax_ckpt_by_metrics = []
        if save_argmin_ckpt_by_metrics is None:
            save_argmin_ckpt_by_metrics = []
        min_metrics, max_metrics = {}, {}

        if os.path.exists(f'{self.workdir}/min_metrics.json'):
            min_metrics = json.load(open(
                f'{self.workdir}/min_metrics.json'))
            self._deployer.log_info(
                json.dumps(min_metrics, indent=4), title='Detected min_metrics')

        if os.path.exists(f'{self.workdir}/max_metrics.json'):
            max_metrics = json.load(open(
                f'{self.workdir}/max_metrics.json'))
            self._deployer.log_info(
                json.dumps(max_metrics, indent=4), title='Detected max_metrics')

        if eval_sanity_check and eval_examples is not None:
            rng_backup = self._deployer._rng
            _, eval_global_micro_batch_size = \
                self._deployer.get_local_global_micro_batch_size(
                    per_device_batch_size=eval_per_device_batch_size)

            if eval_loss:
                self.eval_loss(
                    examples=eval_examples[:eval_global_micro_batch_size],
                    per_device_batch_size=eval_per_device_batch_size,
                    desc=f'Sanity check')
                self._deployer.log_info(
                    'Sanity check (for evaluation loss) passed.')

            if eval_predictor is not None:
                preds = eval_predictor.predict(
                    examples=eval_examples[:eval_global_micro_batch_size],
                    params=self._state.params,
                    params_replicated=(self.mesh is None),
                    params_sharded=(self.mesh is not None),
                    per_device_batch_size=eval_per_device_batch_size,
                    desc=f'Sanity check')
                self._deployer.log_info(
                    'Sanity check (for prediction) passed.')

                if eval_metric_fn is not None:
                    json.dumps(eval_metric_fn(
                        examples=eval_examples[:eval_global_micro_batch_size],
                        preds=preds))
                    self._deployer.log_info(
                        'Sanity check (for evaluation metrics) passed.')

            self._deployer._rng = rng_backup

        for epoch_idx in range(self._init_epoch_idx, n_epochs):
            if isinstance(train_examples, list):
                epoch_train_examples = train_examples
            else:
                epoch_train_examples = train_examples(epoch_idx=epoch_idx)

            self.train(
                examples=epoch_train_examples,
                per_device_batch_size=per_device_batch_size,
                desc=f'epoch {epoch_idx} / {n_epochs}')

            save_ckpt_kwargs = {
                'epoch_idx': epoch_idx,
                'save_opt_state': save_opt_states,
                'float_dtype': save_float_dtype
            }

            if eval_examples is None:
                self._deployer.log_info(
                    'No evaluation cuz \'eval_examples\' is None.')
            else:
                eval_metrics = {}

                if eval_loss:
                    loss = self.eval_loss(
                        examples=eval_examples,
                        per_device_batch_size=eval_per_device_batch_size,
                        desc=f'epoch {epoch_idx} / {n_epochs}')
                    eval_metrics['loss'] = loss

                if eval_predictor is not None:
                    preds = eval_predictor.predict(
                        examples=eval_examples,
                        params=self._state.params,
                        params_replicated=(self.mesh is None),
                        params_sharded=(self.mesh is not None),
                        per_device_batch_size=eval_per_device_batch_size,
                        desc=f'epoch {epoch_idx} / {n_epochs}')

                    if eval_metric_fn is not None:
                        eval_metrics.update(eval_metric_fn(
                            examples=eval_examples, preds=preds))

                    eval_outputs = [
                        {'example': example, 'pred': pred}
                        for example, pred in zip(eval_examples, preds)]

                    self._deployer.save_outputs(
                        outputs=eval_outputs,
                        desc=f'epoch{epoch_idx}',
                        step=self.step)

                self._deployer.log_info(
                    info=json.dumps(eval_metrics, indent=4),
                    title=f'Eval results',
                    step=self.step)
                self._deployer.log_metrics(metrics={
                    f'eval_{key}': value
                    for key, value in eval_metrics.items()
                }, step=self.step)

                if self.workdir is not None:
                    result_filepath = \
                        f'{self.workdir}/eval_results_epoch{epoch_idx}.json'
                    json.dump(
                        eval_metrics, open(result_filepath, 'w'), indent=4)
                    self._deployer.log_info(
                        f'eval_results saved into {result_filepath}.')

                for key in save_argmin_ckpt_by_metrics:
                    assert self.workdir is not None
                    if eval_metrics[key] < min_metrics.get(key, float('inf')):
                        min_metrics[key] = eval_metrics[key]

                        if jax.process_index() == 0:
                            self._deployer.log_info(
                                f'minimal {key} updated to {min_metrics[key]}')
                            json.dump(min_metrics, open(
                                f'{self.workdir}/min_metrics.json', 'w'))

                        self.save_ckpt(
                            ckpt_name=f'min_{key}', **save_ckpt_kwargs)

                for key in save_argmax_ckpt_by_metrics:
                    assert self.workdir is not None
                    if eval_metrics[key] > max_metrics.get(key, float('-inf')):
                        max_metrics[key] = eval_metrics[key]

                        if jax.process_index() == 0:
                            self._deployer.log_info(
                                f'maximal {key} updated to {max_metrics[key]}')
                            json.dump(max_metrics, open(
                                f'{self.workdir}/max_metrics.json', 'w'))

                        self.save_ckpt(
                            ckpt_name=f'max_{key}', **save_ckpt_kwargs)

            if save_every_ckpt:
                self.save_ckpt(
                    ckpt_name=f'epoch_{epoch_idx}', **save_ckpt_kwargs)
            elif save_last_ckpt:
                self.save_ckpt(ckpt_name='last', **save_ckpt_kwargs)

    def save_ckpt(self, epoch_idx, ckpt_name, save_opt_state, float_dtype):
        """Saves a checkpoint into `{self.workdir}/ckpts`.

        Args:
            epoch_idx (int): The current epoch index.
            ckpt_name (str): The name of the checkpoint.
            save_opt_state (bool): Whether to save the optimizer state.
            float_dtype (`jax.numpy.dtype`): Data type for saving float params.
        """
        if self.mesh is None:
            params = jax.tree.map(
                fully_replicated_host_local_array_to_global_array,
                self._state.params)
        else:
            params = self._state.params

        opt_state = None
        if save_opt_state:
            if self.mesh is None:
                opt_state = jax.tree.map(
                    fully_replicated_host_local_array_to_global_array,
                    self._state.opt_state)
            else:
                opt_state = self._state.opt_state

        ckpt_dir = f'{self.workdir}/ckpts/{ckpt_name}'
        self._deployer.save_ckpt(
            ckpt_dir=ckpt_dir,
            params=params,
            opt_state=opt_state,
            float_dtype=float_dtype,
            step=self.step,
            epoch_idx=epoch_idx)

        if jax.process_index() == 0:
            open(f'{self.workdir}/ckpts/last_ckpt.txt', 'w').write(ckpt_name)
            self._deployer.log_info(f'last ckpt updated -- {ckpt_dir}')

    @property
    def step(self):
        """Returns the current training step."""
        if self.mesh is None:
            return unreplicate(self._state.step).item()
        else:
            return self._state.step.item()

    @property
    def workdir(self):
        """Returns the working directory for saving checkpoints and logs."""
        return self._deployer.workdir

    @property
    def mesh(self):
        """Returns the mesh used for distributed training."""
        return self._deployer.mesh

    @property
    def state(self):
        """Returns the current training state."""
        return self._state
