from functools import partial

from .predictor import Predictor
from ..utils.image_to_text_utils import \
    image_to_text_default_collate_fn,\
    image_to_text_default_pred_fn,\
    image_to_text_default_output_fn


class ImageToTextPredictor(Predictor):
    def __init__(self,
                 deployer,
                 model,
                 tokenizer,
                 images_to_pixel_values_fn,
                 max_tgt_len,
                 generation_config,
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

        pred_fn = partial(
            image_to_text_default_pred_fn,
            model=model,
            generation_config=generation_config)

        output_fn = partial(
            image_to_text_default_output_fn, tokenizer=tokenizer)

        super(ImageToTextPredictor, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            pred_fn=pred_fn,
            output_fn=output_fn,
            params=model.params,
            params_shard_rules=params_shard_rules)
