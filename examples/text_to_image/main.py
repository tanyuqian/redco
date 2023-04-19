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

import glob
from PIL import Image
from functools import partial
import os
import fire
import json
import jax
import optax
from diffusers import FlaxStableDiffusionPipeline

from redco import Deployer, Trainer
from text_to_image_pipeline import collate_fn, loss_fn, pred_fn, output_fn


def get_dreambooth_dataset(data_dir,
                           instance_desc,
                           class_desc,
                           prompts,
                           predictor,
                           per_device_batch_size,
                           n_instance_samples_per_epoch,
                           n_class_samples_per_epoch,
                           with_prior_preservation):
    instance_images = []
    for filename in glob.glob(f'{data_dir}/*'):
        instance_images.append(Image.open(filename))

    instance_prompt = \
        prompts['train'].format(instance_or_class_desc=instance_desc)

    dataset = {'train': []}
    for idx in range(n_instance_samples_per_epoch):
        dataset['train'].append({
            'image': instance_images[idx % len(instance_images)],
            'text': instance_prompt
        })

    if with_prior_preservation:
        class_prompt = prompts['train'].format(
            instance_or_class_desc=class_desc)
        examples_to_predict = \
            [{'text': class_prompt}] * n_class_samples_per_epoch

        images = predictor.predict(
            examples=examples_to_predict,
            per_device_batch_size=per_device_batch_size)

        for image in images:
            dataset['train'].append({'image': image, 'text': class_prompt})

    dataset['test'] = [
        {'text': test_prompt.format(instance_desc=instance_desc)}
        for test_prompt in prompts['test']]

    return dataset


def main(data_dir='data/dataset/dog6',
         instance_desc='skr dog',
         class_desc='dog',
         n_instance_samples_per_epoch=400,
         n_class_samples_per_epoch=200,
         model_name_or_path='runwayml/stable-diffusion-v1-5',
         resolution=512,
         n_infer_steps=50,
         n_epochs=6,
         with_prior_preservation=True,
         train_text_encoder=False,
         per_device_batch_size=1,
         eval_per_device_batch_size=2,
         accumulate_grad_batches=1,
         learning_rate=5e-6,
         weight_decay=1e-2,
         jax_seed=42):
    with jax.default_device(jax.devices('cpu')[0]):
        pipeline, pipeline_params = FlaxStableDiffusionPipeline.from_pretrained(
            model_name_or_path, revision="flax")

        params = {}
        for key in list(pipeline_params.keys()):
            if key == 'unet' or (train_text_encoder and key == 'text_encoder'):
                params[key] = pipeline_params.pop(key)

        pipeline_params = pipeline.unet.to_fp16(pipeline_params)
        params = pipeline.unet.to_fp32(params)

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        every_k_schedule=accumulate_grad_batches)

    deployer = Deployer(jax_seed=jax_seed)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn, resolution=resolution, pipeline=pipeline),
        apply_fn=lambda x: None,
        loss_fn=partial(
            loss_fn,
            pipeline=pipeline,
            freezed_params=pipeline_params,
            noise_scheduler_state=pipeline.scheduler.create_state()),
        params=params,
        optimizer=optimizer)

    predictor = trainer.get_default_predictor(
        pred_fn=partial(
            pred_fn,
            pipeline=pipeline,
            freezed_params=pipeline_params,
            n_infer_steps=n_infer_steps,
            resolution=resolution),
        output_fn=partial(output_fn, pipeline=pipeline))

    dataset = get_dreambooth_dataset(
        dataset_name_or_path=dataset_name_or_path,
        prompts=json.load(open('prompts.json')),
        instance_desc=instance_desc,
        class_desc=class_desc,
        predictor=predictor,
        per_device_batch_size=eval_per_device_batch_size,
        n_instance_samples_per_epoch=n_instance_samples_per_epoch,
        n_class_samples_per_epoch=n_class_samples_per_epoch,
        with_prior_preservation=with_prior_preservation)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs)

    images = predictor.predict(
        examples=dataset['test'],
        per_device_batch_size=eval_per_device_batch_size,
        params=trainer.params)

    output_dir = f'{deployer.workdir}/test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    for example, image in zip(dataset['test'], images):
        save_filename = '_'.join(example['text'].split())
        image.save(f'{output_dir}/{save_filename}.jpg')


if __name__ == '__main__':
    fire.Fire(main)