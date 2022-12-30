from functools import partial
import fire
import evaluate

from transformers import \
    AutoImageProcessor, AutoTokenizer, FlaxVisionEncoderDecoderModel

from redco import \
    JsonlDataset, Deployer, ImageToTextTrainer, ImageToTextPredictor


LEARNING_RATE = 1e-5
WARMUP_RATE = 0.1
WEIGHT_DECAY = 0.
JAX_SEED = 42
MAX_TGT_LEN = 16
GEN_KWARGS = {
    'max_length': 16,
    'num_beams': 4
}


def eval_rouge(eval_results, caption_key):
    rouge_scorer = evaluate.load('rouge')

    return rouge_scorer.compute(
        predictions=[result['pred'] for result in eval_results],
        references=[result['example'][caption_key] for result in eval_results],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(data_dir='mscoco_data/processed',
         model_name_or_path='nlpconnect/vit-gpt2-image-captioning',
         n_epochs=2,
         per_device_batch_size=2,
         accumulate_grad_batches=2,
         eval_per_device_batch_size=4,
         image_path_key='image_path',
         caption_key='caption'):
    dataset = JsonlDataset(data_dir=data_dir)

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxVisionEncoderDecoderModel.from_pretrained(
        model_name_or_path, from_pt=True)

    deployer = Deployer(jax_seed=JAX_SEED)

    optimizer = deployer.get_adamw_optimizer(
        train_size=dataset.get_size(split='train'),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=LEARNING_RATE,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=WARMUP_RATE,
        weight_decay=WEIGHT_DECAY)

    trainer = ImageToTextTrainer(
        apply_fn=model.__call__,
        params=model.params,
        optimizer=optimizer,
        deployer=deployer,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_tgt_len=MAX_TGT_LEN,
        image_path_key=image_path_key,
        caption_key=caption_key)

    predictor = ImageToTextPredictor(
        model=model,
        deployer=deployer,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_tgt_len=MAX_TGT_LEN,
        gen_kwargs=GEN_KWARGS,
        image_path_key=image_path_key,
        caption_key=caption_key)

    trainer.fit(
        train_examples=dataset.get_examples('train'),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset.get_examples('dev'),
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=partial(eval_rouge, caption_key=caption_key))


if __name__ == '__main__':
    fire.Fire(main)