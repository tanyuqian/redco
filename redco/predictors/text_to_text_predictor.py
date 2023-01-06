from functools import partial

from .predictor import Predictor
from ..utils.text_to_text_utils import \
    text_to_text_default_collate_fn,\
    text_to_text_default_pred_fn,\
    text_to_text_default_output_fn


class TextToTextPredictor(Predictor):
    def __init__(self,
                 model,
                 deployer,
                 tokenizer,
                 decoder_start_token_id,
                 max_src_len,
                 max_tgt_len,
                 gen_kwargs,
                 dummy_example,
                 src_key='src',
                 tgt_key='tgt',
                 params=None,
                 params_shard_rules=None):
        collate_fn = partial(
            text_to_text_default_collate_fn,
            tokenizer=tokenizer,
            decoder_start_token_id=decoder_start_token_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            src_key=src_key,
            tgt_key=tgt_key)

        pred_fn = partial(
            text_to_text_default_pred_fn, model=model, gen_kwargs=gen_kwargs)
        output_fn = partial(text_to_text_default_output_fn, tokenizer=tokenizer)

        super(TextToTextPredictor, self).__init__(
            deployer=deployer,
            collate_fn=collate_fn,
            pred_fn=pred_fn,
            output_fn=output_fn,
            params=params,
            params_shard_rules=params_shard_rules,
            dummy_example=dummy_example)
