from functools import partial

from .predictor import Predictor
from ..utils.image_to_text_utils import collate_fn, pred_fn, output_fn


class ImageToTextPredictor(Predictor):
    def __init__(self, deployer,
                 model,
                 image_processor,
                 tokenizer,
                 decoder_start_token_id,
                 max_tgt_len,
                 gen_kwargs,
                 image_path_key='image_path',
                 caption_key='caption'):
        super(ImageToTextPredictor, self).__init__(
            deployer=deployer, model=model)

        self.setup_collate_fn(partial(
            collate_fn,
            image_processor=image_processor,
            tokenizer=tokenizer,
            decoder_start_token_id=decoder_start_token_id,
            max_tgt_len=max_tgt_len,
            image_path_key=image_path_key,
            caption_key=caption_key))

        self.setup_pred_step(
            pred_fn=partial(pred_fn, gen_kwargs=gen_kwargs),
            output_fn=partial(output_fn, tokenizer=tokenizer))