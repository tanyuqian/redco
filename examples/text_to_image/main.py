from functools import partial
import fire
import datasets
import evaluate
import jax
import optax

from diffusers import FlaxStableDiffusionPipeline

from redco import Deployer, TextToImageTrainer, TextToImagePredictor


def main(dataset_name='lambdalabs/pokemon-blip-captions',
         image_key='image',
         text_key='text',
         model_name_or_path='flax/stable-diffusion-2-1',
         resolution=512,
         n_infer_steps=50,
         n_epochs=2,
         per_device_batch_size=2,
         eval_per_device_batch_size=4,
         accumulate_grad_batches=2,
         learning_rate=4e-5,
         warmup_rate=0.1,
         weight_decay=1e-2,
         jax_seed=42):
    dataset = {
        'train': list(
            datasets.load_dataset(dataset_name, split='train[:90%]')),
        'validation': list(
            datasets.load_dataset(dataset_name, split='train[90%:]'))
    }

    pipeline, pipeline_params = \
        FlaxStableDiffusionPipeline.from_pretrained(model_name_or_path)

    lr_schedule_fn = optax.constant_schedule(value=learning_rate)
    optimizer = optax.adamw(
        learning_rate=lr_schedule_fn, weight_decay=weight_decay)

    deployer = Deployer(jax_seed=jax_seed, mesh_model_shards=1)

    trainer = TextToImageTrainer(
        deployer=deployer,
        apply_fn=pipeline.unet.apply,
        params=pipeline_params['unet'],
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        pipeline=pipeline,
        pipeline_params=pipeline_params,
        resolution=resolution,
        dummy_example=dataset['train'][0],
        image_key=image_key,
        text_key=text_key,
        params_shard_rules=None)

    predictor = TextToImagePredictor(
        deployer=deployer,
        pipeline=pipeline,
        pipeline_params=pipeline_params,
        resolution=resolution,
        n_infer_steps=n_infer_steps,
        dummy_example=dataset['validation'][0],
        image_key=image_key,
        text_key=text_key)

    preds = predictor.predict(dataset['validation'])
    for i, pred in preds:
        print(i)
        pred.save(f'outputs/{i}.jpg')

    # trainer.fit(
    #     train_examples=dataset['train'],
    #     per_device_batch_size=per_device_batch_size,
    #     n_epochs=n_epochs,
    #     eval_examples=dataset['validation'],
    #     eval_predictor=predictor,
    #     eval_loss=True,
    #     eval_per_device_batch_size=eval_per_device_batch_size)


if __name__ == '__main__':
    fire.Fire(main)