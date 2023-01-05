from functools import partial

from .trainer import Trainer
from ..utils.text_to_text_utils import \
    text_to_text_default_collate_fn, text_to_text_default_loss_fn


class TextToTextTrainer(Trainer):
    def __init__(self,
                 apply_fn,
                 params,
                 optimizer,
                 lr_schedule_fn,
                 deployer,
                 tokenizer,
                 decoder_start_token_id,
                 max_src_len,
                 max_tgt_len,
                 dummy_example,
                 src_key='src',
                 tgt_key='tgt',
                 params_shard_rules=None):
        collate_fn = partial(
            text_to_text_default_collate_fn,
            tokenizer=tokenizer,
            decoder_start_token_id=decoder_start_token_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            src_key=src_key,
            tgt_key=tgt_key)

        super(TextToTextTrainer, self).__init__(
            collate_fn=collate_fn,
            apply_fn=apply_fn,
            loss_fn=text_to_text_default_loss_fn,
            params=params,
            optimizer=optimizer,
            deployer=deployer,
            lr_schedule_fn=lr_schedule_fn,
            dummy_example=dummy_example,
            params_shard_rules=params_shard_rules)
