from functools import partial

import jax

from flax.training.common_utils import shard
from flax.jax_utils import unreplicate

from .utils import TrainStateWithDropoutRNG, default_train_step


class Trainer:
    def __init__(self, deployer, jax_seed):
        self._deployer = deployer

        self._rng = jax.random.PRNGKey(seed=jax_seed)

        self._state = None
        self._data_preprocess_fn = None
        self._p_train_step = None

    def create_train_state(self, apply_fn, params, optimizer):
        self._rng, dropout_rng = jax.random.split(self._rng)

        assert self._deployer.mesh is None
        self._state = TrainStateWithDropoutRNG.create(
            apply_fn=apply_fn,
            params=params,
            tx=optimizer,
            dropout_rng=dropout_rng)

        self._state = self._state.replicate()

    def setup_data_preprocessing(self, data_preprocess_fn):
        self._data_preprocess_fn = data_preprocess_fn

    def setup_train_step(self, loss_fn):
        assert self._deployer.mesh is None
        train_step_fn = partial(default_train_step, loss_fn=loss_fn)
        self._p_train_step = jax.pmap(train_step_fn, axis_name='batch')

    def train_epoch(self, examples):
        self._rng, shuffle_rng = jax.random.split(self._rng)
        examples = self._deployer.get_host_examples(
            examples=examples, shuffle=True, shuffle_rng=shuffle_rng)

        data_batches = self._deployer.get_data_batches(
            examples=examples, desc='Training Epoch')
        for batch in data_batches:
            assert self._deployer.mesh is None

            self._state, metrics = \
                self._p_train_step(state=self._state, batch=shard(batch))
            metrics = unreplicate(metrics)

            data_batches.set_postfix(**metrics)

    def eval(self, examples):
        pass

    def fit(self, ):
        pass