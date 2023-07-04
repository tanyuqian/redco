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

from functools import partial
import json
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from flax.jax_utils import replicate
from flax.training import train_state
from flax.traverse_util import flatten_dict
from flax.core.frozen_dict import freeze
from .utils import default_train_step, default_eval_step
from ..predictors import Predictor


class Trainer:
    def __init__(self,
                 deployer,
                 collate_fn,
                 apply_fn,
                 loss_fn,
                 params,
                 optimizer,
                 lr_schedule_fn=None,
                 params_sharding_rules=None,
                 params_grad_weights=None):
        self._deployer = deployer
        self._collate_fn = collate_fn
        self._loss_fn = loss_fn
        self._lr_schedule_fn = lr_schedule_fn
        self._params_sharding_rules = params_sharding_rules
        self._params_grad_weights = freeze(params_grad_weights) \
            if params_grad_weights is not None else None

        self._state = None
        self._state_spec = None
        self._p_train_step = None
        self._p_eval_step = None

        self.create_train_state(
            apply_fn=apply_fn, params=params, optimizer=optimizer)

        n_params = sum([param.size for param in flatten_dict(params).values()])
        self._deployer.log_info(f'{n_params:,}', title='Training parameters')

        self._default_predictor_fn = partial(
            Predictor,
            deployer=deployer,
            collate_fn=collate_fn,
            params_sharding_rules=params_sharding_rules)

    def create_train_state(self, apply_fn, params, optimizer):
        params = freeze(params)

        if self._deployer.mesh is None:
            self._state = train_state.TrainState.create(
                apply_fn=apply_fn, params=params, tx=optimizer)
            self._state = replicate(self._state)
        else:
            params_spec = self._deployer.get_params_spec(
                params=params,
                params_sharding_rules=self._params_sharding_rules)

            params, opt_state, opt_state_spec = \
                self._deployer.shard_params_and_opt_state(
                    params=params, params_spec=params_spec, optimizer=optimizer)

            self._state = train_state.TrainState(
                apply_fn=apply_fn,
                params=params,
                tx=optimizer,
                opt_state=opt_state,
                step=0)

            self._state_spec = train_state.TrainState(
                apply_fn=apply_fn,
                params=params_spec,
                tx=optimizer,
                opt_state=opt_state_spec,
                step=None)

    def setup_running_step(self, dummy_batch):
        train_step_fn = partial(
            default_train_step,
            loss_fn=self._loss_fn,
            lr_schedule_fn=self._lr_schedule_fn,
            params_grad_weights=self._params_grad_weights,
            under_pmap=(self._deployer.mesh is None))

        eval_step_fn = partial(
            default_eval_step,
            loss_fn=self._loss_fn,
            under_pmap=(self._deployer.mesh is None))

        if self._deployer.mesh is None:
            self._p_train_step = jax.pmap(train_step_fn, axis_name='batch')
            self._p_eval_step = jax.pmap(eval_step_fn, axis_name='batch')
        else:
            data_spec = {
                key: P(*(('dp',) + (None,) * (len(value.shape) - 1)))
                for key, value in dummy_batch.items()
            }

            self._p_train_step = pjit(
                train_step_fn,
                in_axis_resources=(None, self._state_spec, data_spec),
                out_axis_resources=(self._state_spec, None),
                donate_argnums=(1, ))

            self._p_eval_step = pjit(
                eval_step_fn,
                in_axis_resources=(self._state_spec, data_spec),
                out_axis_resources=None)

    def train(self, examples, per_device_batch_size, desc=''):
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            shuffle_rng=self._deployer.gen_rng(),
            desc=f'Training ({desc})')

        for batch in data_batches:
            if self._p_train_step is None:
                self.setup_running_step(dummy_batch=batch)

            train_rng = self._deployer.process_to_run_model(
                self._deployer.gen_rng())
            self._state, metrics = self._deployer.run_model_step(
                step_fn=self._p_train_step,
                input_args=(train_rng, self._state, batch))

            metrics = self._deployer.process_to_deliver(metrics)
            data_batches.set_postfix(**metrics)
            self._deployer.log_metrics(metrics=metrics, step=self.step)

    def eval_loss(self, examples, per_device_batch_size, desc=''):
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            shuffle_rng=None,
            desc=f'Evaluating ({desc})')

        losses = []
        for batch in data_batches:
            if self._p_eval_step is None:
                self.setup_running_step(dummy_batch=batch)

            metrics = self._deployer.run_model_step(
                step_fn=self._p_eval_step, input_args=(self._state, batch))

            metrics = self._deployer.process_to_deliver(metrics)
            data_batches.set_postfix(**metrics)

            losses.append(metrics['loss'].item())

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
            save_every_ckpt=False,
            save_argmin_ckpt_by_metrics=None,
            save_argmax_ckpt_by_metrics=None,
            save_last_ckpt=False):
        if save_argmax_ckpt_by_metrics is None:
            save_argmax_ckpt_by_metrics = []
        if save_argmin_ckpt_by_metrics is None:
            save_argmin_ckpt_by_metrics = []
        min_metrics, max_metrics = {}, {}

        for epoch_idx in range(n_epochs):
            if isinstance(train_examples, list):
                epoch_train_examples = train_examples
            else:
                epoch_train_examples = train_examples(epoch_idx=epoch_idx)

            self.train(
                examples=epoch_train_examples,
                per_device_batch_size=per_device_batch_size,
                desc=f'epoch {epoch_idx} / {n_epochs}')

            if eval_examples is None:
                self._deployer.log_info(
                    'No evaluation cuz \'eval_examples\' is None.')
            else:
                eval_metrics = {}

                if eval_per_device_batch_size is None:
                    eval_per_device_batch_size = per_device_batch_size

                if eval_loss:
                    loss = self.eval_loss(
                        examples=eval_examples,
                        per_device_batch_size=eval_per_device_batch_size,
                        desc=f'epoch {epoch_idx} / {n_epochs}')
                    eval_metrics['loss'] = loss

                if eval_predictor is not None:
                    preds = eval_predictor.predict(
                        examples=eval_examples,
                        params=self.params,
                        params_meshed=(self._deployer.mesh is not None),
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

                if save_every_ckpt:
                    assert self._deployer.workdir is not None
                    path_to_save = f'{self._deployer.workdir}/ckpts/' \
                                   f'epoch_{epoch_idx}.msgpack'
                    self._deployer.save_params(
                        params=self.params,
                        params_sharding_rules=self._params_sharding_rules,
                        filepath=path_to_save)

                if save_last_ckpt:
                    assert self._deployer.workdir is not None
                    path_to_save = f'{self._deployer.workdir}/ckpts/'\
                                   f'last.msgpack'
                    self._deployer.save_params(
                        params=self.params,
                        params_sharding_rules=self._params_sharding_rules,
                        filepath=path_to_save)

                for key in save_argmin_ckpt_by_metrics:
                    assert self._deployer.workdir is not None
                    if eval_metrics[key] < min_metrics.get(key, float('inf')):
                        min_metrics[key] = eval_metrics[key]
                        self._deployer.log_info(
                            f'minimal {key} updated to {min_metrics[key]}')

                        path_to_save = f'{self._deployer.workdir}/ckpts/'\
                                       f'min_{key}.msgpack'
                        self._deployer.save_params(
                            params=self.params,
                            params_sharding_rules=self._params_sharding_rules,
                            filepath=path_to_save)

                for key in save_argmax_ckpt_by_metrics:
                    assert self._deployer.workdir is not None
                    if eval_metrics[key] > max_metrics.get(key, float('-inf')):
                        max_metrics[key] = eval_metrics[key]
                        self._deployer.log_info(
                            f'minimal {key} updated to {max_metrics[key]}')

                        path_to_save = f'{self._deployer.workdir}/ckpts/'\
                                       f'max_{key}.msgpack'
                        self._deployer.save_params(
                            params=self.params,
                            params_sharding_rules=self._params_sharding_rules,
                            filepath=path_to_save)

    def get_default_predictor(self, pred_fn, output_fn=None):
        return self._default_predictor_fn(pred_fn=pred_fn, output_fn=output_fn)

    @property
    def params(self):
        return self._deployer.process_to_deliver(self._state.params)

    @property
    def step(self):
        return self._deployer.process_to_deliver(self._state.step)

