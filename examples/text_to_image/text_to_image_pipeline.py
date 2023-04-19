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

import numpy as np
import jax
import jax.numpy as jnp
from torchvision import transforms


def collate_fn(examples, pipeline, resolution):
    batch = pipeline.tokenizer(
        [example['text'] for example in examples],
        max_length=pipeline.tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    if 'image' in examples[0]:
        images = [example['image'].convert('RGB') for example in examples]

        image_transforms = transforms.Compose([
            transforms.Resize(resolution),
            transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

        batch['pixel_values'] = np.stack(
            [image_transforms(image) for image in images]).astype(np.float16)

    return batch


def loss_fn(train_rng,
            state,
            params,
            batch,
            is_training,
            pipeline,
            freezed_params,
            noise_scheduler_state):
    dropout_rng, sample_rng, noise_rng, timestep_rng = \
        jax.random.split(train_rng, num=4)

    # Convert images to latent space
    vae_outputs = pipeline.vae.apply(
        {"params": freezed_params['vae']},
        batch["pixel_values"],
        deterministic=True,
        method=pipeline.vae.encode)
    latents = vae_outputs.latent_dist.sample(sample_rng)
    latents = jnp.transpose(latents, (0, 3, 1, 2))  # (NHWC) -> (NCHW)
    latents = latents * 0.18215

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)

    # Sample a random timestep for each image
    bsz = latents.shape[0]
    timesteps = jax.random.randint(
        key=timestep_rng,
        shape=(bsz,),
        minval=0,
        maxval=pipeline.scheduler.config.num_train_timesteps)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = pipeline.scheduler.add_noise(
        state=noise_scheduler_state,
        original_samples=latents,
        noise=noise,
        timesteps=timesteps)

    # Get the text embedding for conditioning
    if 'text_encoder' in params:
        encoder_hidden_states = pipeline.text_encoder(
            batch["input_ids"],
            params=params["text_encoder"],
            dropout_rng=dropout_rng,
            train=is_training)[0]
    else:
        encoder_hidden_states = pipeline.text_encoder(
            batch["input_ids"],
            params=freezed_params['text_encoder'],
            train=False)[0]

    # Predict the noise residual
    model_pred = pipeline.unet.apply(
        {"params": params["unet"]},
        sample=noisy_latents,
        timesteps=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        train=is_training).sample

    # Get the target for loss depending on the prediction type
    if pipeline.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif pipeline.scheduler.config.prediction_type == "v_prediction":
        target = pipeline.scheduler.get_velocity(
            state=noise_scheduler_state,
            sample=latents,
            noise=noise,
            timesteps=timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type "
            f"{pipeline.scheduler.config.prediction_type}")

    return jnp.mean(jnp.square(target - model_pred))


def pred_fn(pred_rng,
            batch,
            params,
            pipeline,
            freezed_params,
            n_infer_steps,
            resolution,
            guidance_scale=7.5):
    if 'text_encoder' in params:
        text_encoder_params = params['text_encoder']
    else:
        text_encoder_params = freezed_params['text_encoder']

    pred_params = {
        'unet': params['unet'],
        'text_encoder': text_encoder_params,
        'vae': freezed_params['vae'],
        'scheduler': freezed_params['scheduler']
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
