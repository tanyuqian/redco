from functools import partial

from .trainer import Trainer
from ..utils.text_to_image_utils import \
    text_to_image_default_collate_fn, text_to_image_default_loss_fn

from ..predictors import TextToImagePredictor


class TextToImageTrainer(Trainer):
    def __init__(self,
                 deployer,
                 pipeline,
                 pipeline_params,
                 resolution,
                 optimizer,
                 learning_rate,
                 image_key='image',
                 text_key='text',
                 params_shard_rules=None):
        collate_fn = partial(
            text_to_image_default_collate_fn,
            pipeline=pipeline,
            resolution=resolution,
            image_key=image_key,
            text_key=text_key)

        loss_fn = partial(
            text_to_image_default_loss_fn,
            pipeline=pipeline,
            freezed_params=pipeline_params)

        super(TextToImageTrainer, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            apply_fn=lambda x: None,
            loss_fn=loss_fn,
            params=pipeline_params['unet'],
            optimizer=optimizer,
            learning_rate=learning_rate,
            params_shard_rules=params_shard_rules)

        self._default_predictor_fn = partial(
            TextToImagePredictor,
            deployer=deployer,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            resolution=resolution,
            image_key=image_key,
            text_key=text_key,
            params={'unet': pipeline_params['unet']},
            params_shard_rules=params_shard_rules)

    def get_default_predictor(self, n_infer_steps):
        return self._default_predictor_fn(n_infer_steps=n_infer_steps)
