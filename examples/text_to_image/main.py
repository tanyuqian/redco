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

from functools import partial
import os
import fire
import numpy as np
import jax
import jax.numpy as jnp
import optax
import datasets
from torchvision import transforms
from diffusers import FlaxStableDiffusionPipeline

from redco import Deployer, Trainer, Predictor


def collate_fn(examples, image_key, text_key, resolution, tokenizer):
    batch = tokenizer(
        [example[text_key] for example in examples],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors='np')

    if image_key in examples[0]:
        image_transforms = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

        images = [example[image_key].convert("RGB") for example in examples]
        batch['pixel_values'] = np.stack(
            [image_transforms(image) for image in images]).astype(np.float16)

    return batch


def pred_fn(pred_rng,
            batch,
            params,
            pipeline,
            pipeline_params,
            n_infer_steps,
            guidance_scale):
    resolution = pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    pred_params = {
        'unet': params['unet'],
        'text_encoder': pipeline_params['text_encoder'],
        'vae': pipeline_params['vae'],
        'scheduler': pipeline_params['scheduler']
    }

    return pipeline._generate(
        prompt_ids=batch['input_ids'],
        params=pred_params,
        prng_seed=pred_rng,
        num_inference_steps=n_infer_steps,
        guidance_scale=guidance_scale,
        height=resolution,
        width=resolution)


def output_fn(batch_preds, pipeline):
    return pipeline.numpy_to_pil(np.asarray(batch_preds))


def main(dataset_name='lambdalabs/pokemon-blip-captions',
         image_key='image',
         text_key='text',
         model_name_or_path='duongna/stable-diffusion-v1-4-flax',
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         learning_rate=1e-5,
         grad_norm_clip=1.,
         weight_decay=1e-2,
         n_infer_steps=50,
         guidance_scale=7.5,
         jax_seed=42):
    deployer = Deployer(jax_seed=jax_seed)

    dataset = list(datasets.load_dataset(dataset_name, split='train'))
    cut = int(0.9 * len(dataset))
    dataset = {'train': dataset[:cut], 'test': dataset[cut:]}

    with jax.default_device(jax.devices('cpu')[0]):
        pipeline, pipeline_params = FlaxStableDiffusionPipeline.from_pretrained(
            model_name_or_path, savety_checker=None)
        params = {'unet': pipeline_params.pop('unet')}

        pipeline_params = pipeline.unet.to_fp16(pipeline_params)
        params = pipeline.unet.to_fp32(params)

    # optimizer = optax.MultiSteps(optax.chain(
    #     optax.clip_by_global_norm(grad_norm_clip),
    #     optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    # ), every_k_schedule=accumulate_grad_batches)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            image_key=image_key,
            text_key=text_key,
            resolution=-1,
            tokenizer=pipeline.tokenizer),
        pred_fn=partial(
            pred_fn,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            n_infer_steps=n_infer_steps,
            guidance_scale=guidance_scale),
        output_fn=partial(output_fn, pipeline=pipeline))

    images = predictor.predict(
        examples=dataset['test'],
        params=params,
        per_device_batch_size=eval_per_device_batch_size)

    output_dir = f'{deployer.workdir}/test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    for example, image in zip(dataset['test'], images):
        save_filename = '_'.join(example['text'].split())
        image.save(f'{output_dir}/{save_filename}.jpg')


if __name__ == '__main__':
    fire.Fire(main)