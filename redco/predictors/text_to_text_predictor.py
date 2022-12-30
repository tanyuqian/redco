from functools import partial

from .predictor import Predictor
from ..utils.text_to_text_utils import collate_fn, pred_fn, output_fn


class TextToTextPredictor(Predictor):
    def __init__(self,
                 model,
                 deployer,
                 tokenizer,
                 decoder_start_token_id,
                 max_src_len,
                 max_tgt_len,
                 gen_kwargs,
                 src_key='src',
                 tgt_key='tgt'):
        super(TextToTextPredictor, self).__init__(
            model=model, deployer=deployer)

        self.setup_collate_fn(partial(
            collate_fn,
            tokenizer=tokenizer,
            decoder_start_token_id=decoder_start_token_id,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            src_key=src_key,
            tgt_key=tgt_key))

        self.setup_pred_step(
            pred_fn=partial(pred_fn, gen_kwargs=gen_kwargs),
            output_fn=partial(output_fn, tokenizer=tokenizer))