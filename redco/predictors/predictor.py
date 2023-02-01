from functools import partial

import jax
from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P
from flax.core.frozen_dict import freeze

from .utils import add_idxes, collate_fn_wrapper, pred_fn_wrapper


class Predictor:
    def __init__(self,
                 deployer,
                 collate_fn,
                 pred_fn,
                 output_fn=None,
                 params=None,
                 params_shard_rules=None):
        self._deployer = deployer
        self._collate_fn = partial(collate_fn_wrapper, collate_fn=collate_fn)

        self._params = freeze(params) if params is not None else None
        self._params_shard_rules = params_shard_rules
        self._pred_fn = partial(
            pred_fn_wrapper,
            pred_fn=pred_fn,
            under_pmap=self._deployer.mesh is None)
        self._p_pred_step = None

        if output_fn is None:
            self._output_fn = lambda x: x.tolist()
        else:
            self._output_fn = output_fn

    def setup_running_step(self, pred_fn, dummy_batch, params_shard_rules):
        if self._deployer.mesh is None:
            self._p_pred_step = jax.pmap(pred_fn, axis_name='batch')
        else:
            data_spec = {
                key: P(*(('dp',) + (None,) * (len(value.shape) - 1)))
                for key, value in dummy_batch.items()
            }

            params_spec = self._deployer.get_params_spec(
                params=self._params, shard_rules=params_shard_rules)

            self._p_pred_step = pjit(
                pred_fn,
                in_axis_resources=(None, params_spec, data_spec),
                out_axis_resources=None)

    def predict(self, examples, per_device_batch_size, params=None):
        if params is None:
            params = self._params

        raw_n_inputs = len(examples)
        _, global_batch_size = self._deployer.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        examples = examples + [examples[0]] * (global_batch_size - 1)
        examples = add_idxes(examples=examples)

        params = self._deployer.process_to_run_model(params)

        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            shuffle_rng=None,
            desc='Predicting')

        preds = []
        for batch in data_batches:
            if self._p_pred_step is None:
                self.setup_running_step(
                    pred_fn=self._pred_fn,
                    dummy_batch=batch,
                    params_shard_rules=self._params_shard_rules)

            pred_rng = self._deployer.process_to_run_model(
                self._deployer.gen_rng(), is_prng_key=True)

            batch_preds_with_idxes = self._deployer.run_model_step(
                step_fn=self._p_pred_step,
                input_args=(pred_rng, params, batch))

            batch_preds = self._deployer.process_batch_preds(
                batch_preds_with_idxes=batch_preds_with_idxes)
            batch_preds = self._output_fn(batch_preds)
            preds.extend(batch_preds)

        return preds[:raw_n_inputs]