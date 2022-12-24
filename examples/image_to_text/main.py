import fire

import optax

from transformers import \
    AutoImageProcessor, AutoTokenizer, FlaxVisionEncoderDecoderModel

from redco import JsonlDataset, Deployer, ImageToTextTrainer


def main(data_dir='mscoco_data/processed',
         model_name_or_path='nlpconnect/vit-gpt2-image-captioning',
         workdir='./outputs_image_to_text',
         max_tgt_len=16,
         n_epochs=2,
         learning_rate=1e-5,
         per_device_batch_size=2,
         jax_seed=42):
    dataset = JsonlDataset(data_dir=data_dir)

    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxVisionEncoderDecoderModel.from_pretrained(
        model_name_or_path, from_pt=True)

    deployer = Deployer(workdir=workdir)

    trainer = ImageToTextTrainer(
        deployer=deployer,
        image_processor=image_processor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_tgt_len=max_tgt_len)

    optimizer = optax.adam(learning_rate=learning_rate)
    trainer.create_train_state(
        apply_fn=model.__call__,
        params=model.params,
        optimizer=optimizer,
        jax_seed=jax_seed)

    trainer.fit(
        train_examples=dataset.get_examples(split='train'),
        train_per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs)


if __name__ == '__main__':
    fire.Fire(main)