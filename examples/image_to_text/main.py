import json

import fire

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

    deployer = Deployer(workdir=workdir)

    predictor = ImageToTextPredictor(
        deployer=deployer,
        model=model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_tgt_len=MAX_TGT_LEN,
        gen_kwargs=GEN_KWARGS)

    preds = predictor.predict(
        params=model.params,
        examples=dataset.get_examples(split='test'),
        per_device_batch_size=per_device_batch_size)

    results = [
        {'example': example, 'pred': pred}
        for example, pred in zip(dataset.get_examples(split='test'), preds)]

    json.dump(results, open('results.json', 'w'))

    # trainer = ImageToTextTrainer(
    #     deployer=deployer,
    #     image_processor=image_processor,
    #     tokenizer=tokenizer,
    #     decoder_start_token_id=model.config.decoder_start_token_id,
    #     max_tgt_len=MAX_TGT_LEN)
    #
    # optimizer = deployer.get_adamw_optimizer(
    #     train_size=dataset.get_size(split='train'),
    #     per_device_batch_size=per_device_batch_size,
    #     n_epochs=n_epochs,
    #     learning_rate=LEARNING_RATE,
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     warmup_rate=WARMUP_RATE,
    #     weight_decay=WEIGHT_DECAY)
    # trainer.create_train_state(
    #     apply_fn=model.__call__,
    #     params=model.params,
    #     optimizer=optimizer,
    #     jax_seed=JAX_SEED)
    #
    # trainer.fit(
    #     train_examples=dataset.get_examples(split='train'),
    #     train_per_device_batch_size=per_device_batch_size,
    #     n_epochs=n_epochs)


if __name__ == '__main__':
    fire.Fire(main)