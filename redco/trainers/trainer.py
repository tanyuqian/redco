from functools import partial

import numpy as np
import jax

from .utils import \
    TrainStateWithDropoutRNG, default_train_step, default_eval_step


class Trainer:
    def __init__(self, deployer):
        self._deployer = deployer

        self._rng = None
        self._state = None
        self._data_preprocess_fn = None
        self._p_train_step = None
        self._p_eval_step = None

    def create_train_state(self,
                           apply_fn,
                           params,
                           optimizer,
                           jax_seed):
        self._rng = jax.random.PRNGKey(seed=jax_seed)
        self._rng, dropout_rng = jax.random.split(self._rng)

        assert self._deployer.mesh is None
        self._state = TrainStateWithDropoutRNG.create(
            apply_fn=apply_fn,
            params=params,
            tx=optimizer,
            dropout_rng=dropout_rng)

        self._state = self._state.replicate()

    def setup_step_fns(self, loss_fn):
        assert self._deployer.mesh is None
        train_step_fn = partial(default_train_step, loss_fn=loss_fn)
        self._p_train_step = jax.pmap(train_step_fn, axis_name='batch')
        eval_step_fn = partial(default_eval_step, loss_fn=loss_fn)
        self._p_eval_step = jax.pmap(eval_step_fn, axis_name='batch')

    def train_epoch(self, examples, per_device_batch_size, epoch_idx):
        self._rng, shuffle_rng = jax.random.split(self._rng)
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            data_preprocess_fn=self._data_preprocess_fn,
            shuffle=True,
            shuffle_rng=shuffle_rng,
            desc=f'Training epoch {epoch_idx}')

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
            data_preprocess_fn=self._data_preprocess_fn,
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

        return np.mean(losses)