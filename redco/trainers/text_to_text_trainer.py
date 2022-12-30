from functools import partial

from .trainer import Trainer
from ..utils.text_to_text_utils import collate_fn, loss_fn


class TextToTextTrainer(Trainer):
    def __init__(self,
                 apply_fn,
                 params,
                 optimizer,
                 deployer,
                 tokenizer,
                 decoder_start_token_id,
                 max_src_len,
                 max_tgt_len,
                 src_key='src',
                 tgt_key='tgt'):
        super(TextToTextTrainer, self).__init__(
            apply_fn=apply_fn,
            params=params,
            optimizer=optimizer,
            deployer=deployer)

        self._collate_fn = partial(
            collate_fn,
            tokenizer=tokenizer,
            decoder_start_token_id=decoder_start_token_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            src_key=src_key,
            tgt_key=tgt_key)

        self.setup_loss_fn(loss_fn=loss_fn)
