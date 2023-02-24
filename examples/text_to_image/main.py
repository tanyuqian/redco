from functools import partial
import os
import fire
import jax
import optax
from diffusers import FlaxStableDiffusionPipeline

from redco import Deployer, Trainer
from text_to_image_pipeline import (
    text_to_image_collate_fn,
    text_to_image_loss_fn,
    text_to_image_pred_fn,
    text_to_image_output_fn)

from data_utils import get_dreambooth_dataset


def main(instance_desc='skr dog',
         class_desc='dog',
         n_instance_samples_per_epoch=400,
         n_class_samples_per_epoch=200,
         image_key='image',
         text_key='text',
         model_name_or_path='runwayml/stable-diffusion-v1-5',
         resolution=512,
         n_infer_steps=50,
         n_epochs=8,
         with_prior_preservation=True,
         train_text_encoder=False,
         per_device_batch_size=1,
         eval_per_device_batch_size=1,
         accumulate_grad_batches=1,
         learning_rate=5e-6,
         weight_decay=1e-2,
         output_dir='outputs',
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

    collate_fn = partial(
        text_to_image_collate_fn,
        resolution=resolution,
        pipeline=pipeline,
        image_key=image_key,
        text_key=text_key)

    loss_fn = partial(
        text_to_image_loss_fn,
        pipeline=pipeline,
        freezed_params=pipeline_params,
        noise_scheduler_state=pipeline.scheduler.create_state())

    pred_fn = partial(
        text_to_image_pred_fn,
        pipeline=pipeline,
        freezed_params=pipeline_params,
        n_infer_steps=n_infer_steps,
        resolution=resolution)

    output_fn = partial(text_to_image_output_fn, pipeline=pipeline)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=lambda x: None,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer)

    predictor = trainer.get_default_predictor(
        pred_fn=pred_fn, output_fn=output_fn)

    dataset = get_dreambooth_dataset(
        predictor=predictor,
        per_device_batch_size=eval_per_device_batch_size,
        instance_desc=instance_desc,
        class_desc=class_desc,
        n_instance_samples_per_epoch=n_instance_samples_per_epoch,
        n_class_samples_per_epoch=n_class_samples_per_epoch,
        text_key=text_key,
        image_key=image_key,
        with_prior_preservation=with_prior_preservation)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs)

    images = predictor.predict(
        examples=dataset['validation'],
        per_device_batch_size=eval_per_device_batch_size,
        params=trainer.params)

    os.makedirs(output_dir, exist_ok=True)
    for example, image in zip(dataset['validation'], images):
        save_filename = '_'.join(example[text_key].split())
        image.save(f'{output_dir}/{save_filename}.jpg')


if __name__ == '__main__':
    fire.Fire(main)