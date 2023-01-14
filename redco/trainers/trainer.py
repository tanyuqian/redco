from functools import partial
import json
import pickle
import numpy as np

import jax
from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P

from .utils import \
    TrainState, default_train_step, default_eval_step, get_lr_schedule_fn


class Trainer:
    def __init__(self,
                 deployer,
                 collate_fn,
                 apply_fn,
                 loss_fn,
                 params,
                 optimizer,
                 learning_rate,
                 params_shard_rules=None):
        self._deployer = deployer
        self._collate_fn = collate_fn

        self._state = None
        self._state_spec = None

        self.create_train_state(
            apply_fn=apply_fn,
            params=params,
            params_shard_rules=params_shard_rules,
            optimizer=optimizer,
            lr_schedule_fn=get_lr_schedule_fn(learning_rate=learning_rate))

        self._loss_fn = loss_fn
        self._p_train_step = None
        self._p_eval_step = None

    def create_train_state(self,
                           apply_fn,
                           params,
                           params_shard_rules,
                           optimizer,
                           lr_schedule_fn):
        if self._deployer.mesh is None:
            self._state = TrainState.create(
                apply_fn=apply_fn,
                params=params,
                tx=optimizer,
                train_rng=self._deployer.gen_rng(),
                lr_schedule_fn=lr_schedule_fn)

            self._state = self._state.replicate()
        else:
            params_spec = self._deployer.get_params_spec(
                params=params, shard_rules=params_shard_rules)

            params, opt_state, opt_state_spec = \
                self._deployer.shard_params_and_opt_state(
                    params=params, params_spec=params_spec, optimizer=optimizer)

            self._state = TrainState(
                apply_fn=apply_fn,
                params=params,
                tx=optimizer,
                opt_state=opt_state,
                train_rng=self._deployer.gen_rng(),
                lr_schedule_fn=lr_schedule_fn,
                step=0)

            self._state_spec = TrainState(
                apply_fn=apply_fn,
                params=params_spec,
                tx=optimizer,
                opt_state=opt_state_spec,
                train_rng=None,
                lr_schedule_fn=lr_schedule_fn,
                step=None)

    def setup_running_step(self, loss_fn, dummy_batch):
        # print('Batch shapes (\"-1\" -> batch_size):')
        # print(json.dumps(jax.tree_util.tree_map(
        #     lambda x: (-1, ) + tuple(x.shape[1:]), dummy_batch)))

        if self._deployer.mesh is None:
            self._p_train_step = jax.pmap(partial(
                default_train_step, loss_fn=loss_fn, under_pmap=True),
                axis_name='batch')
            self._p_eval_step = jax.pmap(partial(
                default_eval_step, loss_fn=loss_fn, under_pmap=True),
                axis_name='batch')
        else:
            data_spec = {
                key: P(*(('dp',) + (None,) * (len(value.shape) - 1)))
                for key, value in dummy_batch.items()
            }

            self._p_train_step = pjit(
                partial(default_train_step, loss_fn=loss_fn, under_pmap=False),
                in_axis_resources=(self._state_spec, data_spec),
                out_axis_resources=(self._state_spec, None),
                donate_argnums=(0,))

            self._p_eval_step = pjit(
                partial(default_eval_step, loss_fn=loss_fn, under_pmap=False),
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
                self.setup_running_step(
                    loss_fn=self._loss_fn, dummy_batch=batch)

            self._state, metrics = self._deployer.run_model_step(
                step_fn=self._p_train_step, input_args=(self._state, batch))

            metrics = self._deployer.process_to_deliver(metrics)
            data_batches.set_postfix(**metrics)

    def eval_loss(self, examples, per_device_batch_size):
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            shuffle_rng=None,
            desc=f'Evaluating')

        losses = []
        for batch in data_batches:
            if self._p_eval_step is None:
                self.setup_running_step(
                    loss_fn=self._loss_fn, dummy_batch=batch)

            metrics = self._deployer.run_model_step(
                step_fn=self._p_eval_step, input_args=(self._state, batch))

            metrics = self._deployer.process_to_deliver(metrics)
            data_batches.set_postfix(**metrics)

            losses.append(metrics['loss'])

        return np.mean(losses).item()

    def fit(self,
            train_examples,
            per_device_batch_size,
            n_epochs,
            eval_examples=None,
            eval_per_device_batch_size=None,
            eval_loss=True,
            eval_predictor=None,
            eval_metric_fn=None):
        for epoch_idx in range(n_epochs):
            if isinstance(train_examples, list):
                epoch_train_examples = train_examples
            else:
                epoch_train_examples = train_examples(epoch_idx=epoch_idx)

            self.train(
                examples=epoch_train_examples,
                per_device_batch_size=per_device_batch_size,
                desc=f'epoch {epoch_idx}')

            if eval_examples is None:
                pass
                # print('No evaluation cuz \'eval_examples\' is None.')
            else:
                eval_metrics = {}

                if eval_per_device_batch_size is None:
                    eval_per_device_batch_size = per_device_batch_size

                if eval_loss:
                    loss = self.eval_loss(
                        examples=eval_examples,
                        per_device_batch_size=eval_per_device_batch_size)
                    eval_metrics['loss'] = loss

                if eval_predictor is not None:
                    preds = eval_predictor.predict(
                        examples=eval_examples,
                        params=self.params,
                        per_device_batch_size=eval_per_device_batch_size)

                    eval_results = [
                        {'example': example, 'pred': pred}
                        for example, pred in zip(eval_examples, preds)]

                    try:
                        json.dump(
                            eval_results,
                            open(f'outputs_epoch{epoch_idx}.json', 'w'),
                            indent=4)
                    except:
                        pickle.dump(eval_results, open(
                            f'outputs_epoch{epoch_idx}.pkl', 'wb'))

                    if eval_metric_fn is not None:
                        eval_metrics.update(eval_metric_fn(eval_results))

                print(f'Epoch {epoch_idx}, evaluation results:')
                print(json.dumps(eval_metrics, indent=4))

    @property
    def params(self):
        return self._deployer.process_to_deliver(self._state.params)

    @property
    def step(self):
        return self._deployer.process_to_deliver(self._state.step)
