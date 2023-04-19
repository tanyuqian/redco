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
import datasets
from transformers import \
    AutoImageProcessor, AutoTokenizer, FlaxVisionEncoderDecoderModel

from redco import Deployer, Trainer
from image_to_text_pipeline import (
    image_to_text_collate_fn,
    image_to_text_default_loss_fn,
    image_to_text_default_pred_fn,
    image_to_text_default_output_fn,
    eval_rouge)


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

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    collate_fn = partial(
        image_to_text_collate_fn,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_tgt_len=max_tgt_len,
        image_path_key=image_path_key,
        text_key=text_key)

    pred_fn = partial(
        image_to_text_default_pred_fn, model=model, gen_kwargs=gen_kwargs)

    output_fn = partial(image_to_text_default_output_fn, tokenizer=tokenizer)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model,
        loss_fn=image_to_text_default_loss_fn,
        params=model.params,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn)

    predictor = trainer.get_default_predictor(
        pred_fn=pred_fn, output_fn=output_fn)

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
