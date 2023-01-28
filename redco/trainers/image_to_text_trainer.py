from functools import partial

from .trainer import Trainer
from ..utils.image_to_text_utils import \
    image_to_text_default_collate_fn, image_to_text_default_loss_fn

from ..predictors import ImageToTextPredictor


class ImageToTextTrainer(Trainer):
    def __init__(self,
                 deployer,
                 model,
                 optimizer,
                 lr_schedule_fn,
                 tokenizer,
                 images_to_pixel_values_fn,
                 max_tgt_len,
                 image_path_key='image_path',
                 image_key=None,
                 text_key='caption',
                 params_shard_rules=None):
        collate_fn = partial(
            image_to_text_default_collate_fn,
            images_to_pixel_values_fn=images_to_pixel_values_fn,
            tokenizer=tokenizer,
            decoder_start_token_id=model.config.decoder_start_token_id,
            max_tgt_len=max_tgt_len,
            image_path_key=image_path_key,
            image_key=image_key,
            text_key=text_key)

        super(ImageToTextTrainer, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            apply_fn=model.__call__,
            loss_fn=image_to_text_default_loss_fn,
            params=model.params,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            params_shard_rules=params_shard_rules)

        self._default_predictor_fn = partial(
            ImageToTextPredictor,
            deployer=deployer,
            model=model,
            tokenizer=tokenizer,
            images_to_pixel_values_fn=images_to_pixel_values_fn,
            max_tgt_len=max_tgt_len,
            image_path_key=image_path_key,
            image_key=image_key,
            text_key=text_key,
            params_shard_rules=params_shard_rules)

    def get_default_predictor(self, generation_config):
        return self._default_predictor_fn(generation_config=generation_config)
