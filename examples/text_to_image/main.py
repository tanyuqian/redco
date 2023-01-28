from functools import partial
import os
import fire
import jax
import optax
import numpy as np

from torchvision import transforms
from diffusers import FlaxStableDiffusionPipeline

from redco import Deployer, TextToImageTrainer

from dreambooth_utils import get_dreambooth_dataset


def images_to_pixel_values_fn(images, image_transforms):
    return np.stack(
        [image_transforms(image) for image in images],
        dtype=np.float16)


def main(instance_dir='./skr_dog_images',
         instance_desc='skr dog',
         class_dir='./normal_dog_images',
         class_desc='dog',
         n_instance_samples_per_epoch=400,
         n_class_samples_per_epoch=200,
         image_key='image',
         image_path_key=None,
         text_key='text',
         model_name_or_path='duongna/stable-diffusion-v1-4-flax',
         resolution=512,
         n_infer_steps=50,
         n_epochs=8,
         with_prior_preservation=False,
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
            model_name_or_path)

        params = {}
        for key in list(pipeline_params.keys()):
            if key == 'unet' or (train_text_encoder and key == 'text_encoder'):
                params[key] = pipeline_params.pop(key)

        pipeline_params = pipeline.unet.to_fp16(pipeline_params)
        params = pipeline.unet.to_fp32(params)

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        every_k_schedule=accumulate_grad_batches)

    image_transforms = transforms.Compose([
        transforms.Resize(resolution),
        transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    deployer = Deployer(jax_seed=jax_seed)

    trainer = TextToImageTrainer(
        deployer=deployer,
        pipeline=pipeline,
        params=params,
        freezed_params=pipeline_params,
        resolution=resolution,
        optimizer=optimizer,
        images_to_pixel_values_fn=partial(
            images_to_pixel_values_fn, image_transforms=image_transforms),
        image_key=image_key,
        image_path_key=image_path_key,
        text_key=text_key,
        params_shard_rules=None)

    predictor = trainer.get_default_predictor(n_infer_steps=n_infer_steps)

    dataset = get_dreambooth_dataset(
        predictor=predictor,
        per_device_batch_size=eval_per_device_batch_size,
        instance_dir=instance_dir,
        instance_desc=instance_desc,
        class_dir=class_dir,
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