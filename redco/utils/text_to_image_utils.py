import numpy as np
import jax
import jax.numpy as jnp

import PIL.Image


def preprocess(image, resolution, dtype):
    image = image.convert('RGB').resize((resolution, resolution))
    image = np.array(image) / 255.0
    image = image.transpose(2, 0, 1)
    return image


def text_to_image_default_collate_fn(examples,
                                     pipeline,
                                     resolution,
                                     image_key='image',
                                     text_key='text'):
    batch = pipeline.tokenizer(
        [example[text_key] for example in examples],
        max_length=pipeline.tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    batch['pixel_values'] = np.stack([
        preprocess(example['image'], resolution=resolution, dtype=np.float32)
        for example in examples])

    return batch


def text_to_image_default_loss_fn(
        state, params, batch, train, pipeline, pipeline_params):
    # Convert images to latent space
    vae_outputs = pipeline.vae.apply(
        {"params": pipeline_params['vae']},
        batch["pixel_values"],
        deterministic=True,
        method=pipeline.vae.encode)
    latents = vae_outputs.latent_dist.sample(sample_rng)

    # (NHWC) -> (NCHW)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    latents = latents * 0.18215

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(state.dropout_rng)
    noise = jax.random.normal(noise_rng, latents.shape)

    # Sample a random timestep for each image
    bsz = latents.shape[0]
    timesteps = jax.random.randint(
        key=timestep_rng,
        shape=(bsz,),
        minval=0,
        maxval=noise_scheduler.config.num_train_timesteps)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(
        batch["input_ids"],
        params=pipeline_params['text_encoder'],
        train=False)[0]

    # Predict the noise residual and compute loss
    unet_outputs = state.apply_fn(
        {"params": params},
        noisy_latents, timesteps, encoder_hidden_states,
        train=True)
    noise_pred = unet_outputs.sample
    loss = (noise - noise_pred) ** 2
    loss = loss.mean()

    return loss