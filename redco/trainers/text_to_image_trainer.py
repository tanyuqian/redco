from functools import partial

from .trainer import Trainer
from ..utils.text_to_image_utils import \
    text_to_image_default_collate_fn, text_to_image_default_loss_fn

from ..predictors import TextToImagePredictor


class TextToImageTrainer(Trainer):
    def __init__(self,
                 deployer,
                 pipeline,
                 params,
                 freezed_params,
                 resolution,
                 optimizer,
                 image_key='image',
                 text_key='text',
                 coscum_image_preprocess_fn=None,
                 lr_schedule_fn=None,
                 params_shard_rules=None):
        collate_fn = partial(
            text_to_image_default_collate_fn,
            pipeline=pipeline,
            resolution=resolution,
            coscum_image_preprocess_fn=coscum_image_preprocess_fn,
            image_key=image_key,
            text_key=text_key)

        loss_fn = partial(
            text_to_image_default_loss_fn,
            pipeline=pipeline,
            freezed_params=freezed_params)

        super(TextToImageTrainer, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            apply_fn=lambda x: None,
            loss_fn=loss_fn,
            params=params,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            params_shard_rules=params_shard_rules)

        self._default_predictor_fn = partial(
            TextToImagePredictor,
            deployer=deployer,
            pipeline=pipeline,
            freezed_params=freezed_params,
            resolution=resolution,
            image_key=image_key,
            text_key=text_key,
            params=params,
            params_shard_rules=params_shard_rules)

    def get_default_predictor(self, n_infer_steps):
        return self._default_predictor_fn(n_infer_steps=n_infer_steps)
