import numpy as np
from PIL import Image

import jax.numpy as jnp
import optax


def preprocess(example,
               image_processor,
               tokenizer,
               decoder_start_token_id,
               max_tgt_len,
               image_path_key='image_path',
               caption_key='caption'):
    model_inputs = {}

    img = Image.open(example[image_path_key]).convert('RGB')
    model_inputs['pixel_values'] = \
        image_processor(img, return_tensors='np').pixel_values[0]

    decoder_inputs = tokenizer(
        example[caption_key],
        add_special_tokens=False,
        max_length=max_tgt_len,
        padding='max_length',
        truncation=True)

    model_inputs['labels'] = decoder_inputs['input_ids']
    decoder_inputs['input_ids'] = \
        [decoder_start_token_id] + decoder_inputs['input_ids'][:-1]

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


def pred_step(batch, params, model, gen_kwargs):
    output_ids = model.generate(
        batch["pixel_values"], params=params, **gen_kwargs)
    return output_ids.sequences


def postprocess(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)