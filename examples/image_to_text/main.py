import os
from functools import partial
import fire

import datasets
import evaluate
from transformers import \
    AutoImageProcessor,\
    AutoTokenizer,\
    FlaxVisionEncoderDecoderModel,\
    GenerationConfig

from redco import Deployer, ImageToTextTrainer


def images_to_pixel_values_fn(images, image_processor):
    return image_processor(images, return_tensors='np').pixel_values


def eval_rouge(eval_results, text_key):
    rouge_scorer = evaluate.load('rouge')

    return rouge_scorer.compute(
        predictions=[result['pred'] for result in eval_results],
        references=[result['example'][text_key] for result in eval_results],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(data_dir='./mscoco_data',
         image_path_key='image_path',
         image_key=None,
         text_key='caption',
         model_name_or_path='nlpconnect/vit-gpt2-image-captioning',
         n_epochs=2,
         per_device_batch_size=8,
         accumulate_grad_batches=2,
         eval_per_device_batch_size=16,
         learning_rate=1e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         jax_seed=42,
         max_tgt_len=16,
         num_beams=4):
    dataset = datasets.load_dataset(
        "ydshieh/coco_dataset_script", "2017",
        data_dir=os.path.abspath(f'{data_dir}/raw'),
        cache_dir=f'{data_dir}/cache')
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxVisionEncoderDecoderModel.from_pretrained(
        model_name_or_path, from_pt=True)
    try:
        generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    except:
        generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(max_length=max_tgt_len, num_beams=num_beams)

    deployer = Deployer(jax_seed=jax_seed)

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    trainer = ImageToTextTrainer(
        deployer=deployer,
        model=model,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        tokenizer=tokenizer,
        images_to_pixel_values_fn=partial(
            images_to_pixel_values_fn, image_processor=image_processor),
        max_tgt_len=max_tgt_len,
        image_path_key=image_path_key,
        image_key=image_key,
        text_key=text_key)

    predictor = trainer.get_default_predictor(
        generation_config=generation_config)

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=partial(eval_rouge, text_key=text_key))


if __name__ == '__main__':
    fire.Fire(main)
