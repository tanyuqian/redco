from functools import partial

from .trainer import Trainer
from ..utils.image_to_text_utils import preprocess, loss_fn


class ImageToTextTrainer(Trainer):
    def __init__(self,
                 deployer,
                 image_processor,
                 tokenizer,
                 decoder_start_token_id,
                 max_tgt_len,
                 image_path_key='image_path',
                 caption_key='caption'):
        super(ImageToTextTrainer, self).__init__(deployer=deployer)

        self._data_preprocess_fn = partial(
            preprocess,
            image_processor=image_processor,
            tokenizer=tokenizer,
            decoder_start_token_id=decoder_start_token_id,
            max_tgt_len=max_tgt_len,
            image_path_key=image_path_key,
            caption_key=caption_key)

        self.setup_step_fns(loss_fn=loss_fn)
