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
from transformers import CLIPTextConfig, CLIPTokenizer, FlaxCLIPTextModel
from diffusers import (
    FlaxAutoencoderKL,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel)
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


def loss_fn(train_rng,
            state,
            params,
            batch,
            is_training,
            vae,
            noise_scheduler,
            noise_scheduler_state,
            text_encoder):
    sample_rng, noise_rng, timestep_rng = jax.random.split(train_rng, num=3)

    vae_outputs = vae.apply(
        {"params": params['vae']},
        batch["pixel_values"],
        deterministic=True,
        method=vae.encode)
    latents = vae_outputs.latent_dist.sample(sample_rng)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    latents = latents * vae.config.scaling_factor

    noise = jax.random.normal(noise_rng, latents.shape)
    timesteps = jax.random.randint(
        key=timestep_rng,
        shape=(latents.shape[0],),
        minval=0,
        maxval=noise_scheduler.config.num_train_timesteps)
    noisy_latents = noise_scheduler.add_noise(
        state=noise_scheduler_state,
        original_samples=latents,
        noise=noise,
        timesteps=timesteps)

    encoder_hidden_states = text_encoder(
        batch['input_ids'], params=params['text_encoder'], train=False)[0]

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
         n_epochs=8,
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
        config=CLIPTextConfig.from_pretrained(
            model_name_or_path, subfolder="text_encoder"),
        dtype=jnp.float32, _do_init=False)
    vae = FlaxAutoencoderKL.from_config(
        config=FlaxAutoencoderKL.load_config(
            model_name_or_path, subfolder='vae'), dtype=jnp.float32)
    unet = FlaxUNet2DConditionModel.from_config(
        config=FlaxUNet2DConditionModel.load_config(
            model_name_or_path, subfolder='unet'), dtype=jnp.float32)
    noise_scheduler, noise_scheduler_state = \
        FlaxPNDMScheduler.from_pretrained(
            model_name_or_path, subfolder='scheduler')
    pipeline = FlaxStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=None,
        safety_checker=None)

    accumulate_grad_batches = deployer.get_accumulate_grad_batches(
        global_batch_size=global_batch_size,
        per_device_batch_size=per_device_batch_size)
    optimizer = optax.MultiSteps(optax.chain(
        optax.clip_by_global_norm(grad_norm_clip),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    ), every_k_schedule=accumulate_grad_batches)
    param_labels = path_aware_map(  # Only Unet parameters are trainable.
        lambda path, _: 'trainable' if path[0] == 'unet' else 'frozen',
        deployer.load_params_shape(ckpt_dir=init_ckpt_dir))
    optimizer = optax.multi_transform(
        transforms={'trainable': optimizer, 'frozen': optax.set_to_zero()},
        param_labels=freeze(param_labels))

    ckpt, info = deployer.load_last_ckpt(
        optimizer=optimizer, float_dtype=jnp.float32)
    if ckpt is None:
        ckpt, info = deployer.load_ckpt(
            ckpt_dir=init_ckpt_dir, update_rng=False, float_dtype=jnp.float32)

    params_sharding_rules = {}
    for key in ['text_encoder', 'unet', 'vae']:
        params_sharding_rules[key] = deployer.get_sharding_rules(
            params_shape_or_params=ckpt['params'][key])
        if n_model_shards > 1:
            deployer.log_info(
                info='\n'.join([str(t) for t in params_sharding_rules[key]]),
                title=f'Sharding Rules ({key})')

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
        n_epochs=n_epochs,
        save_last_ckpt=False)

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