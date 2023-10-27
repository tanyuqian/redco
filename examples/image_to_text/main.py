#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from functools import partial
import fire
from PIL import Image
import numpy as np
import jax.numpy as jnp
import optax
import datasets
from transformers import (
    AutoImageProcessor, AutoTokenizer, FlaxVisionEncoderDecoderModel)
import evaluate
from redco import Deployer, Trainer


def collate_fn(examples,
               image_processor,
               tokenizer,
               decoder_start_token_id,
               max_tgt_len,
               image_path_key='image_path',
               text_key='caption'):
    images = [
        Image.open(example[image_path_key]).convert('RGB')
        for example in examples
    ]
    model_inputs = {
        'pixel_values': image_processor(
            images, return_tensors='np').pixel_values
    }

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


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop("labels")
    label_weights = batch['decoder_attention_mask']
    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training)[0]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(pred_rng, batch, params, model, gen_kwargs):
    return model.generate(
        batch["pixel_values"], params=params, prng_key=pred_rng, **gen_kwargs
    ).sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)


def eval_rouge(examples, preds, text_key):
    rouge_scorer = evaluate.load('rouge')

    return rouge_scorer.compute(
        predictions=preds,
        references=[example[text_key] for example in examples],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(data_dir='./mscoco_data',
         image_path_key='image_path',
         text_key='caption',
         model_name_or_path='nlpconnect/vit-gpt2-image-captioning',
         n_epochs=2,
         per_device_batch_size=8,
         accumulate_grad_batches=2,
         eval_per_device_batch_size=16,
         learning_rate=1e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         jax_seed=42,
         max_tgt_len=16,
         num_beams=4):
    dataset = datasets.load_dataset(
        "ydshieh/coco_dataset_script", "2017",
        data_dir=os.path.abspath(f'{data_dir}/raw'),
        cache_dir=f'{data_dir}/cache')
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxVisionEncoderDecoderModel.from_pretrained(
        model_name_or_path, from_pt=True)
    gen_kwargs = {'max_length': max_tgt_len, 'num_beams': num_beams}

    deployer = Deployer(jax_seed=jax_seed)

    lr_schedule_fn = deployer.get_lr_schedule_fn(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        schedule_type='linear',
        warmup_rate=warmup_rate)
    optimizer = optax.adamw(
        learning_rate=lr_schedule_fn, weight_decay=weight_decay)
    if accumulate_grad_batches > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=accumulate_grad_batches)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            image_processor=image_processor,
            tokenizer=tokenizer,
            decoder_start_token_id=model.config.decoder_start_token_id,
            max_tgt_len=max_tgt_len,
            image_path_key=image_path_key,
            text_key=text_key),
        apply_fn=model,
        loss_fn=loss_fn,
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn)

    predictor = trainer.get_default_predictor(
        pred_fn=partial(pred_fn, model=model, gen_kwargs=gen_kwargs),
        output_fn=partial(output_fn, tokenizer=tokenizer))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=partial(eval_rouge, text_key=text_key))


if __name__ == '__main__':
    fire.Fire(main)
