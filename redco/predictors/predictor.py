from functools import partial

import jax


class Predictor:
    def __init__(self, deployer):
        self._deployer = deployer

        self._model = None
        self._collate_fn = None
        self._p_pred_step = None
        self._output_fn = None

    def setup_collate_fn(self, collate_fn):
        self._collate_fn = collate_fn

    def setup_pred_step(self, pred_fn, output_fn):
        assert self._deployer.mesh is None
        self._p_pred_step = jax.pmap(partial(
            pred_fn, model=self._model), axis_name='batch')
        self._output_fn = output_fn

    def setup(self, model, collate_fn, pred_fn, output_fn):
        self._model = model
        self.setup_collate_fn(collate_fn=collate_fn)
        self.setup_pred_step(pred_fn=pred_fn, output_fn=output_fn)

    def predict(self, params, examples, per_device_batch_size):
        raw_n_inputs = len(examples)
        _, global_batch_size = self._deployer.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        examples = examples + examples[:global_batch_size - 1]

        params = self._deployer.process_params_to_run(params=params)

        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            collate_fn=self._collate_fn,
            shuffle=False,
            shuffle_rng=None,
            desc='Predicting')

        preds = []
        for batch in data_batches:
            assert self._deployer.mesh is None
            batch_preds = self._p_pred_step(batch=batch, params=params)
            batch_preds = self._deployer.process_batch_preds(
                batch_preds=batch_preds)

            batch_preds = self._output_fn(batch_preds)
            preds.extend(batch_preds)

        return preds[:raw_n_inputs]