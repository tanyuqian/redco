from functools import partial

from .trainer import Trainer
from ..utils.text_to_image_utils import \
    text_to_image_default_collate_fn, text_to_image_default_loss_fn

from diffusers import FlaxStableDiffusionPipeline

class TextToImageTrainer(Trainer):
    def __init__(self,
                 deployer,
                 apply_fn,
                 params,
                 optimizer,
                 lr_schedule_fn,
                 pipeline,
                 pipeline_params,
                 dummy_example,
                 resolution,
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
            pipeline_params=pipeline_params)

        super(TextToImageTrainer, self).__init__(
            collate_fn=collate_fn,
            apply_fn=apply_fn,
            loss_fn=loss_fn,
            params=params,
            optimizer=optimizer,
            deployer=deployer,
            lr_schedule_fn=lr_schedule_fn,
            dummy_example=dummy_example,
            params_shard_rules=params_shard_rules)
