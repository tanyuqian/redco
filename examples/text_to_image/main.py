import os
import fire
import jax
import optax

from diffusers import FlaxStableDiffusionPipeline

from redco import Deployer, TextToImageTrainer

from dreambooth_utils import get_dreambooth_dataset


def main(instance_dir='./skr_dog_images',
         instance_prompt='sks dog',
         class_dir='./normal_dog_images',
         class_prompt='a photo of a dog',
         n_class_images=200,
         image_key='image',
         text_key='text',
         model_name_or_path='flax/stable-diffusion-2-1-base',
         resolution=512,
         n_infer_steps=50,
         n_epochs=1,
         per_device_batch_size=2,
         eval_per_device_batch_size=2,
         accumulate_grad_batches=2,
         learning_rate=5e-6,
         weight_decay=1e-2,
         output_dir='outputs',
         jax_seed=42):
    with jax.default_device(jax.devices('cpu')[0]):
        pipeline, pipeline_params = FlaxStableDiffusionPipeline.from_pretrained(model_name_or_path)
        pipeline_params = pipeline.unet.to_fp16(pipeline_params)
        pipeline_params['unet'] = pipeline.unet.to_fp32(pipeline_params['unet'])

    lr_schedule_fn = optax.constant_schedule(value=learning_rate)
    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule_fn, weight_decay=weight_decay),
        every_k_schedule=accumulate_grad_batches)

    deployer = Deployer(jax_seed=jax_seed, mesh_model_shards=1)

    trainer = TextToImageTrainer(
        deployer=deployer,
        pipeline=pipeline,
        pipeline_params=pipeline_params,
        resolution=resolution,
        optimizer=optimizer,
        learning_rate=lr_schedule_fn,
        image_key=image_key,
        text_key=text_key,
        params_shard_rules=None)

    predictor = trainer.get_default_predictor(n_infer_steps=n_infer_steps)

    dataset = get_dreambooth_dataset(
        predictor=predictor,
        per_device_batch_size=eval_per_device_batch_size,
        instance_dir=instance_dir,
        instance_prompt=instance_prompt,
        class_dir=class_dir,
        class_prompt=class_prompt,
        n_class_images=n_class_images,
        text_key=text_key,
        image_key=image_key)

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