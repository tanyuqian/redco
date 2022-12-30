from functools import partial

import numpy as np
import jax

from .utils import \
    TrainStateWithDropoutRNG, default_train_step, default_eval_step


class Trainer:
    def __init__(self, apply_fn, params, optimizer, deployer):
        self._deployer = deployer

        self._state = None
        self.create_train_state(
            apply_fn=apply_fn, params=params, optimizer=optimizer)

        self._collate_fn = None
        self._p_train_step = None
        self._p_eval_step = None

    def create_train_state(self, apply_fn, params, optimizer):
        assert self._deployer.mesh is None
        self._state = TrainStateWithDropoutRNG.create(
            apply_fn=apply_fn,
            params=params,
            tx=optimizer,
            dropout_rng=self._deployer.gen_rng())

        self._state = self._state.replicate()

    def setup_collate_fn(self, collate_fn):
        self._collate_fn = collate_fn

    def setup_loss_fn(self, loss_fn):
        assert self._deployer.mesh is None
        train_step_fn = partial(default_train_step, loss_fn=loss_fn)
        self._p_train_step = jax.pmap(train_step_fn, axis_name='batch')
        eval_step_fn = partial(default_eval_step, loss_fn=loss_fn)
        self._p_eval_step = jax.pmap(eval_step_fn, axis_name='batch')

    def train(self, examples, per_device_batch_size, desc):
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=True,
            shuffle_rng=self._deployer.gen_rng(),
            desc=f'Training ({desc})')

        for batch in data_batches:
            assert self._deployer.mesh is None

            self._state, metrics = self._p_train_step(
                state=self._state, batch=batch)
            metrics = self._deployer.process_metrics(metrics=metrics)

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
            assert self._deployer.mesh is None

            metrics = self._p_eval_step(state=self._state, batch=batch)
            metrics = self._deployer.process_metrics(metrics=metrics)
            data_batches.set_postfix(**metrics)

            losses.append(metrics['loss'])

        return {'loss': np.mean(losses)}

    def setup(self, collate_fn=None, loss_fn=None):
        if collate_fn is not None:
            self.setup_collate_fn(collate_fn=collate_fn)
        assert self._collate_fn is not None

        if loss_fn is not None:
            self.setup_loss_fn(loss_fn=loss_fn)
        assert self._p_train_step is not None

    def fit(self,
            train_examples,
            per_device_batch_size,
            n_epochs,
            eval_examples=None,
            eval_fn=None):
        for epoch_idx in range(n_epochs):
            self.train(
                examples=train_examples,
                per_device_batch_size=per_device_batch_size,
                desc=f'epoch {epoch_idx}')

            if eval_fn is None:
                eval_result = self.eval_loss(
                    examples=eval_examples,
                    per_device_batch_size=per_device_batch_size_eval)
            else:
                eval_result = eval_fn(params=self.params)

            print(f'Epoch {epoch_idx}, eval result = {eval_result}')

    @property
    def params(self):
        return self._deployer.process_params_to_save(self._state.params)
