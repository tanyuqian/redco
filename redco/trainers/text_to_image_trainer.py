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
                 optimizer,
                 images_to_pixel_values_fn,
                 image_key='image',
                 image_path_key=None,
                 text_key='text',
                 lr_schedule_fn=None,
                 params_shard_rules=None):
        collate_fn = partial(
            text_to_image_default_collate_fn,
            pipeline=pipeline,
            images_to_pixel_values_fn=images_to_pixel_values_fn,
            image_key=image_key,
            image_path_key=image_path_key,
            text_key=text_key)

        loss_fn = partial(
            text_to_image_default_loss_fn,
            pipeline=pipeline,
            freezed_params=freezed_params,
            noise_scheduler_state=pipeline.scheduler.create_state())

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
            text_key=text_key,
            params=params,
            params_shard_rules=params_shard_rules)

    def get_default_predictor(self, resolution, n_infer_steps):
        return self._default_predictor_fn(
            resolution=resolution, n_infer_steps=n_infer_steps)
