from functools import partial

import jax


class Predictor:
    def __init__(self, deployer, model):
        self._deployer = deployer
        self._model = model

        self._data_preprocess_fn = None
        self._p_pred_step = None
        self._postprocess_fn = None

    def setup_pred_step(self, pred_step_fn, postprocess_fn):
        assert self._deployer.mesh is None
        self._p_pred_step = jax.pmap(partial(
            pred_step_fn, model=self._model), axis_name='batch')
        self._postprocess_fn = postprocess_fn

    def predict(self, params, examples, per_device_batch_size):
        data_batches = self._deployer.get_model_input_batches(
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            data_preprocess_fn=self._data_preprocess_fn,
            shuffle=False,
            shuffle_rng=None,
            desc='Predicting')

        preds = []
        for batch in data_batches:
            assert self._deployer.mesh is None
            batch_preds = self._p_pred_step(batch=batch, params=params)
            batch_preds = self._deployer.process_batch_preds(
                batch_preds=batch_preds)

            batch_preds = self._postprocess_fn(batch_preds)
            preds.extend(batch_preds)

        return preds