from functools import partial
import json
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


def eval_rouge(params, predictor, examples, per_device_batch_size, scorer):
    preds = predictor.predict(
        params=params,
        examples=examples,
        per_device_batch_size=per_device_batch_size)

    refs = [example['caption'] for example in examples]

    result = rouge_scorer.compute(
        predictions=preds, references=refs, use_stemmer=True)

    return results


def main(data_dir='mscoco_data/processed',
         model_name_or_path='nlpconnect/vit-gpt2-image-captioning',
         workdir='./outputs_image_to_text',
         n_epochs=2,
         per_device_batch_size=2,
         accumulate_grad_batches=2):
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
        max_tgt_len=MAX_TGT_LEN)

    predictor = ImageToTextPredictor(
        model=model,
        deployer=deployer,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_tgt_len=MAX_TGT_LEN,
        gen_kwargs=GEN_KWARGS)

    scorer = evaluate.load('rouge')

    eval_fn = partial(
        eval_rouge,
        predictor=predictor,
        examples=dataset.get_examples('dev'),
        per_device_batch_size=per_device_batch_size,
        scorer=scorer)

    trainer.fit(
        train_examples=dataset.get_examples(split='train'),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_fn=eval_fn)


if __name__ == '__main__':
    fire.Fire(main)