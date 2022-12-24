from functools import partial

import numpy as np
from PIL import Image

import jax.numpy as jnp
import optax

from .trainer import Trainer


class ImageToTextTrainer(Trainer):
    def __init__(self,
                 deployer,
                 jax_seed,
                 image_processor,
                 tokenizer,
                 decoder_start_token_id,
                 max_tgt_len,
                 image_path_key='image_path',
                 caption_key='caption'):
        super(ImageToTextTrainer, self).__init__(
            deployer=deployer, jax_seed=jax_seed)

        self.setup_data_preprocessing(
            data_preprocess_fn=partial(
                preprocess,
                image_processor=image_processor,
                tokenizer=tokenizer,
                decoder_start_token_id=decoder_start_token_id,
                max_tgt_len=max_tgt_len,
                image_path_key=image_path_key,
                caption_key=caption_key))
        self.setup_train_step(loss_fn=loss_fn)


def preprocess(example,
               image_processor,
               tokenizer,
               decoder_start_token_id,
               max_tgt_len,
               image_path_key='image_path',
               caption_key='caption'):
    model_inputs = {}

    img = Image.open(example[image_path_key])
    model_inputs['pixel_values'] = \
        image_processor(img, return_tensors='np').pixel_values[0]

    decoder_inputs = tokenizer(
        example[caption_key],
        add_special_tokens=False,
        max_length=max_tgt_len,
        padding='max_length',
        truncation=True)

    model_inputs['labels'] = decoder_inputs['input_ids']
    model_inputs['input_ids'] = \
        [decoder_start_token_id] + decoder_inputs['input_ids']

    for key in decoder_inputs:
        model_inputs[f'decoder_{key}'] = np.array(decoder_inputs[key])

    return model_inputs


def loss_fn(state, params, batch, train):
    labels = batch.pop("labels")
    label_weights = batch['decoder_attention_mask']

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=state.dropout_rng, train=train)[0]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


