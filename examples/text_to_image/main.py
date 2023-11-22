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
from transformers import CLIPImageProcessor, CLIPTokenizer, FlaxCLIPTextModel
from diffusers import (
    FlaxAutoencoderKL,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel)
from redco import Deployer, Trainer


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


def loss_fn(train_rng,
            state,
            params,
            batch,
            is_training,
            frozen_params,
            vae,
            noise_scheduler,
            noise_scheduler_state,
            text_encoder):
    vae_outputs = vae.apply(
        {"params": frozen_params['vae']},
        batch["pixel_values"],
        deterministic=True,
        method=vae.encode)

    train_rng, sample_rng = jax.random.split(train_rng)
    latents = vae_outputs.latent_dist.sample(sample_rng)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    latents = latents * vae.config.scaling_factor

    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)
    timesteps = jax.random.randint(
        key=timestep_rng,
        shape=(latents.shape[0],),
        minval=0,
        maxval=noise_scheduler.config.num_train_timesteps)

    noisy_latents = noise_scheduler.add_noise(
        noise_scheduler_state, latents, noise, timesteps)

    encoder_hidden_states = text_encoder(
        batch['input_ids'], params=frozen_params['text_encoder'], train=False
    )[0]

    model_pred = state.apply_fn(
        {"params": params},
        sample=noisy_latents,
        timestpes=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        train=True
    ).sample

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    else:
        assert noise_scheduler.config.prediction_type == "v_prediction"
        target = noise_scheduler.get_velocity(
            noise_scheduler_state, latents, noise, timesteps)

    return jnp.mean((target - model_pred) ** 2)


def pred_fn(pred_rng,
            batch,
            params,
            pipeline,
            frozen_params,
            resolution,
            n_infer_steps,
            guidance_scale):
    pred_params = {
        'unet': params['unet'],
        'text_encoder': frozen_params['text_encoder'],
        'vae': frozen_params['vae'],
        'scheduler': frozen_params['scheduler']
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
         n_epochs=1,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         learning_rate=1e-5,
         grad_norm_clip=1.,
         weight_decay=1e-2,
         n_infer_steps=50,
         guidance_scale=7.5,
         workdir='./workdir',
         jax_seed=42):
    deployer = Deployer(workdir=workdir, jax_seed=jax_seed)

    dataset = list(datasets.load_dataset(dataset_name, split='train'))
    cut = int(0.9 * len(dataset))
    dataset = {'train': dataset[:cut], 'test': dataset[cut:]}

    with jax.default_device(jax.devices('cpu')[0]):
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name_or_path, subfolder="tokenizer")
        text_encoder = FlaxCLIPTextModel.from_pretrained(
            model_name_or_path, subfolder="text_encoder",)
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            model_name_or_path, subfolder="vae")
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            model_name_or_path, subfolder="unet")
        feature_extracter = CLIPImageProcessor.from_pretrained(
            model_name_or_path, subfolder='feature_extractor')
        noise_scheduler, noise_scheduler_state = FlaxPNDMScheduler.\
            from_pretrained(model_name_or_path, subfolder='scheduler')

        pipeline = FlaxStableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            feature_extractor=feature_extracter,
            safety_checker=None)

        params = {'unet': unet_params}
        frozen_params = {
            'text_encoder': text_encoder.params,
            'vae': vae_params,
            'scheduler': noise_scheduler_state
        }

        pipeline_params = pipeline.unet.to_fp16(frozen_params)
        params = pipeline.unet.to_fp32(params)

    optimizer = optax.MultiSteps(optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    ), every_k_schedule=accumulate_grad_batches)

    resolution = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            image_key=image_key,
            text_key=text_key,
            resolution=resolution,
            tokenizer=pipeline.tokenizer),
        apply_fn=pipeline.unet.apply,
        loss_fn=partial(
            loss_fn,
            pipeline_params=pipeline_params,
            vae=pipeline.vae,
            noise_scheduler=pipeline.noise_scheduler,
            noise_scheduler_state=pipeline.noise_scheduler.create,
            text_encoder=text_encoder),
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lambda step: learning_rate,
        accumulate_grad_batches=accumulate_grad_batches)

    predictor = trainer.get_default_predictor(
        pred_fn=partial(
            pred_fn,
            pipeline=pipeline,
            pipeline_params=pipeline_params,
            resolution=resolution,
            n_infer_steps=n_infer_steps,
            guidance_scale=guidance_scale),
        output_fn=partial(output_fn, pipeline=pipeline))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs)

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