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
from flax.core.frozen_dict import freeze
from flax.traverse_util import path_aware_map
import optax
import datasets
from torchvision import transforms
from transformers import CLIPConfig, CLIPTokenizer, FlaxCLIPTextModel
from diffusers import (
    FlaxAutoencoderKL,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel)
from redco import Deployer, Trainer, Predictor


def get_optimizer(deployer,
                  params_shape_or_params,
                  global_batch_size,
                  per_device_batch_size,
                  grad_norm_clip,
                  learning_rate,
                  weight_decay):
    _, global_micro_batch_size = deployer.get_local_global_micro_batch_size(
        per_device_batch_size=per_device_batch_size)
    assert global_batch_size % global_micro_batch_size == 0
    accumulate_grad_batches = global_batch_size // global_micro_batch_size
    deployer.log_info(accumulate_grad_batches, title='accumulate_grad_batches')

    optimizer = optax.MultiSteps(optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    ), every_k_schedule=accumulate_grad_batches)

    # Grouping parameters -- Only Unet parameters are trainable.
    param_labels = path_aware_map(
        lambda path, _: 'trainable' if path[0] == 'unet' else 'frozen',
        params_shape_or_params)
    optimizer = optax.multi_transform(
        transforms={'trainable': optimizer, 'frozen': optax.set_to_zero()},
        param_labels=freeze(param_labels))

    return optimizer, accumulate_grad_batches


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
            vae,
            noise_scheduler,
            noise_scheduler_state,
            text_encoder):
    vae_outputs = vae.apply(
        {"params": params['vae']},
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
        batch['input_ids'], params=params['text_encoder'], train=False
    )[0]

    model_pred = state.apply_fn(
        {"params": params['unet']},
        sample=noisy_latents,
        timesteps=timesteps,
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
            resolution,
            n_infer_steps,
            guidance_scale,
            noise_scheduler_state):
    return pipeline._generate(
        prompt_ids=batch['input_ids'],
        params={**params, 'scheduler': noise_scheduler_state},
        prng_seed=pred_rng,
        num_inference_steps=n_infer_steps,
        guidance_scale=guidance_scale,
        height=resolution,
        width=resolution)


def output_fn(batch_preds, pipeline):
    return pipeline.numpy_to_pil(np.asarray(batch_preds))


def main(dataset_name='lambdalabs/naruto-blip-captions',
         image_key='image',
         text_key='text',
         model_name_or_path='stabilityai/stable-diffusion-2-1-base',
         init_ckpt_dir='./stable-diffusion-2-1-base',
         n_model_shards=1,
         n_epochs=3,
         global_batch_size=8,
         per_device_batch_size=1,
         learning_rate=1e-5,
         grad_norm_clip=1.,
         weight_decay=1e-2,
         n_infer_steps=50,
         guidance_scale=7.5,
         workdir='./workdir',
         jax_seed=42,
         n_processes=None,
         host0_address=None,
         host0_port=None,
         process_id=None,
         n_local_devices=None):
    deployer = Deployer(
        workdir=workdir,
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        n_processes=n_processes,
        host0_address=host0_address,
        host0_port=host0_port,
        process_id=process_id,
        n_local_devices=n_local_devices)

    dataset = list(datasets.load_dataset(dataset_name, split='train'))
    cut = int(0.9 * len(dataset))
    dataset = {'train': dataset[:cut], 'test': dataset[cut:]}

    tokenizer = CLIPTokenizer.from_pretrained(
        model_name_or_path, subfolder="tokenizer")
    text_encoder = FlaxCLIPTextModel(
        config=CLIPConfig.from_pretrained(
            model_name_or_path, subfolder="text_encoder"),
        dtype=jnp.float16, _do_init=False)
    vae = FlaxAutoencoderKL.from_config(
        config=FlaxAutoencoderKL.load_config(
            model_name_or_path, subfolder='vae'), dtype=jnp.float16)
    unet = FlaxUNet2DConditionModel.from_config(
        config=FlaxUNet2DConditionModel.load_config(
            model_name_or_path, subfolder='unet'), dtype=jnp.float16)
    noise_scheduler, noise_scheduler_state = \
        FlaxPNDMScheduler.from_pretrained(
            model_name_or_path, subfolder='scheduler')
    params_shape = {
        'text_encoder': jax.eval_shape(
            partial(text_encoder.init_weights, input_shape=(1, 1)),
            jax.random.PRNGKey(0)),
        'vae': jax.eval_shape(vae.init_weights, jax.random.PRNGKey(0)),
        'unet': jax.eval_shape(unet.init_weights, jax.random.PRNGKey(0))
    }
    params_sharding_rules = {
        key: deployer.get_sharding_rules(
            params_shape_or_params=params_shape[key])
        for key in ['unet', 'text_encoder', 'vae']
    }
    optimizer, accumulate_grad_batches = get_optimizer(
        deployer=deployer,
        params_shape_or_params=params_shape,
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size,
        grad_norm_clip=grad_norm_clip,
        learning_rate=learning_rate,
        weight_decay=weight_decay)

    load_ckpt_kwargs = {
        'params_shape_or_params': params_shape,
        'optimizer': optimizer,
        'params_sharding_rules': params_sharding_rules,
        'float_dypte': jnp.float32
    }
    ckpt, info = deployer.load_last_ckpt(**load_ckpt_kwargs)
    if ckpt is None:
        ckpt, info = deployer.load_ckpt(
            ckpt_dir=init_ckpt_dir, **load_ckpt_kwargs)

    pipeline = FlaxStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None)

    resolution = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    deployer.log_info(resolution, title='resolution')

    collate_fn_kwargs = {
        'image_key': image_key,
        'text_key': text_key,
        'resolution': resolution,
        'tokenizer': pipeline.tokenizer
    }
    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        apply_fn=pipeline.unet.apply,
        loss_fn=partial(
            loss_fn,
            vae=vae,
            noise_scheduler=noise_scheduler,
            noise_scheduler_state=noise_scheduler_state,
            text_encoder=text_encoder),
        params=ckpt['params'],
        opt_state=ckpt['opt_state'],
        last_ckpt_info=info,
        optimizer=optimizer,
        lr_schedule_fn=lambda step: learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        params_sharding_rules=params_sharding_rules)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=partial(collate_fn, **collate_fn_kwargs),
        pred_fn=partial(
            pred_fn,
            pipeline=pipeline,
            resolution=resolution,
            n_infer_steps=n_infer_steps,
            guidance_scale=guidance_scale,
            noise_scheduler_state=noise_scheduler_state),
        output_fn=partial(output_fn, pipeline=pipeline),
        params_sharding_rules=params_sharding_rules)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs)

    images = predictor.predict(
        examples=dataset['test'],
        params=trainer.state.params,
        params_replicated=(trainer.mesh is None),
        params_sharded=(trainer.mesh is not None),
        per_device_batch_size=per_device_batch_size)

    output_dir = f'{deployer.workdir}/test_outputs'
    os.makedirs(output_dir, exist_ok=True)
    for example, image in zip(dataset['test'], images):
        save_filename = ''.join(
            [ch if ch.isalpha() else '_' for ch in example[text_key]])
        image.save(f'{output_dir}/{save_filename}.jpg')


if __name__ == '__main__':
    fire.Fire(main)