import numpy as np
from PIL import Image

import jax.numpy as jnp
import optax


def image_to_text_default_collate_fn(examples,
                                     images_to_pixel_values_fn,
                                     tokenizer,
                                     decoder_start_token_id,
                                     max_tgt_len,
                                     image_path_key='image_path',
                                     image_key=None,
                                     text_key='caption'):
    if image_key is not None:
        images = [example[image_key].convert('RGB') for example in examples]
    else:
        images = [
            Image.open(example[image_path_key]).convert('RGB')
            for example in examples]
    model_inputs = {'pixel_values': images_to_pixel_values_fn(images)}

    decoder_inputs = tokenizer(
        [example[text_key] for example in examples],
        max_length=max_tgt_len,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    if tokenizer.bos_token_id is not None:
        labels = np.zeros_like(decoder_inputs['input_ids'])
        labels[:, :-1] = decoder_inputs['input_ids'][:, 1:]
        decoder_input_ids = decoder_inputs['input_ids']
        decoder_input_ids[:, 0] = decoder_start_token_id
    else:
        labels = decoder_inputs['input_ids']
        decoder_input_ids = np.zeros_like(decoder_inputs['input_ids'])
        decoder_input_ids[:, 1:] = decoder_inputs['input_ids'][:, :-1]
        decoder_input_ids[:, 0] = decoder_start_token_id

    model_inputs['labels'] = labels
    decoder_inputs['input_ids'] = decoder_input_ids

    for key in decoder_inputs:
        model_inputs[f'decoder_{key}'] = np.array(decoder_inputs[key])

    return model_inputs


def image_to_text_default_loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop("labels")
    label_weights = batch['decoder_attention_mask']

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def image_to_text_default_pred_fn(
        pred_rng, batch, params, model, generation_config):
    return model.generate(
        batch["pixel_values"],
        params=params,
        prng_key=pred_rng,
        generation_config=generation_config).sequences


def image_to_text_default_output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)