from functools import partial

from .predictor import Predictor
from ..utils.text_to_image_utils import (
    text_to_image_default_collate_fn,
    text_to_image_default_pred_fn,
    text_to_image_default_output_fn)


class TextToImagePredictor(Predictor):
    def __init__(self,
                 deployer,
                 pipeline,
                 freezed_params,
                 resolution,
                 n_infer_steps,
                 text_key='text',
                 params=None,
                 params_shard_rules=None):
        collate_fn = partial(
            text_to_image_default_collate_fn,
            pipeline=pipeline,
            resolution=resolution,
            text_key=text_key)

        pred_fn = partial(
            text_to_image_default_pred_fn,
            pipeline=pipeline,
            freezed_params=freezed_params,
            n_infer_steps=n_infer_steps,
            resolution=resolution)

        output_fn = partial(
            text_to_image_default_output_fn, pipeline=pipeline)

        super(TextToImagePredictor, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            pred_fn=pred_fn,
            output_fn=output_fn,
            params=params,
            params_shard_rules=params_shard_rules)
