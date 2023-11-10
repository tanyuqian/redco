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
import numpy as np
import jax
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from flax.core.frozen_dict import freeze
from .utils import \
    add_idxes, collate_fn_wrapper, pred_fn_wrapper, default_output_fn


class Predictor:
    def __init__(self,
                 deployer,
                 collate_fn,
                 pred_fn,
                 output_fn=None,
                 params_sharding_rules=None):
        self._deployer = deployer
        self._collate_fn = partial(collate_fn_wrapper, collate_fn=collate_fn)

        self._params_sharding_rules = params_sharding_rules
        self._pred_fn = partial(
            pred_fn_wrapper,
            pred_fn=pred_fn,
            under_pmap=self._deployer.mesh is None)
        self._params_spec = None
        self._p_pred_step = None

        if output_fn is None:
            self._output_fn = default_output_fn
        else:
            self._output_fn = output_fn

    def setup_running_step(self,
                           pred_fn,
                           dummy_batch,
                           params,
                           params_sharding_rules):
        if self._deployer.mesh is None:
            self._p_pred_step = jax.pmap(pred_fn, axis_name='batch')
        else:
            data_spec = jax.tree_util.tree_map(
                lambda x: P(*(('dp',) + (None,) * (len(x.shape) - 1))),
                dummy_batch)

            self._params_spec = self._deployer.get_params_spec(
                params=params, params_sharding_rules=params_sharding_rules)

            self._p_pred_step = pjit(
                pred_fn,
                in_shardings=(None, self._params_spec, data_spec),
                out_shardings=None)

    def predict(self,
                examples,
                per_device_batch_size,
                params,
                params_meshed=False,
                desc=None):
        params = freeze(params)

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
            desc=f'Predicting ({desc})' if desc is not None else 'Predicting')

        preds = []
        for batch in data_batches:
            if self._p_pred_step is None:
                self.setup_running_step(
                    pred_fn=self._pred_fn,
                    dummy_batch=batch,
                    params=params,
                    params_sharding_rules=self._params_sharding_rules)

            if (self._params_spec is not None) and (not params_meshed):
                params = self._deployer.shard_params(
                    params=params, params_spec=self._params_spec)
                params_meshed = True

            pred_rng = self._deployer.process_to_run_model(
                self._deployer.gen_rng(), is_prng_key=True)

            batch_preds_with_idxes = self._deployer.run_model_step(
                step_fn=self._p_pred_step,
                input_args=(pred_rng, params, batch))

            batch_preds = self._deployer.process_batch_preds(
                batch_preds_with_idxes=batch_preds_with_idxes)
            batch_preds = jax.tree_util.tree_map(np.asarray, batch_preds)

            batch_preds = self._output_fn(batch_preds)
            assert isinstance(batch_preds, list) and \
                   len(batch_preds) == global_batch_size
            preds.extend(batch_preds)

        return preds[:raw_n_inputs]