from functools import partial
import fire
import datasets
import evaluate

from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM

from redco import Deployer, TextToTextTrainer, TextToTextPredictor


LEARNING_RATE = 5e-5
WARMUP_RATE = 0.1
WEIGHT_DECAY = 0.
JAX_SEED = 42
MAX_SRC_LEN = 512
MAX_TGT_LEN = 64
GEN_KWARGS = {
    'max_length': 64,
    'num_beams': 4
}


def eval_rouge(params,
               trainer,
               predictor,
               examples,
               per_device_batch_size,
               rouge_scorer):
    loss = trainer.eval_loss(
        examples=examples, per_device_batch_size=per_device_batch_size)['loss']

    preds = predictor.predict(
        params=params,
        examples=examples,
        per_device_batch_size=per_device_batch_size)

    refs = [example['caption'] for example in examples]

    result = rouge_scorer.compute(
        predictions=preds, references=refs, use_stemmer=True)

    result.update({'loss': loss})

    return result


def main(dataset_name='xsum',
         src_key='document',
         tgt_key='summary',
         model_name_or_path='facebook/bart-base',
         n_epochs=2,
         per_device_batch_size=8,
         accumulate_grad_batches=2):
    dataset = datasets.load_dataset(dataset_name)
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, from_pt=True)

    deployer = Deployer(jax_seed=JAX_SEED)

    optimizer = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=LEARNING_RATE,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=WARMUP_RATE,
        weight_decay=WEIGHT_DECAY)

    trainer = TextToTextTrainer(
        apply_fn=model.__call__,
        params=model.params,
        optimizer=optimizer,
        deployer=deployer,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_src_len=MAX_SRC_LEN,
        max_tgt_len=MAX_TGT_LEN,
        src_key=src_key,
        tgt_key=tgt_key)

    predictor = TextToTextPredictor(
        model=model,
        deployer=deployer,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_src_len=MAX_SRC_LEN,
        max_tgt_len=MAX_TGT_LEN,
        gen_kwargs=GEN_KWARGS,
        src_key=src_key,
        tgt_key=tgt_key)

    eval_fn = partial(
        eval_rouge,
        predictor=predictor,
        examples=dataset['test'],
        per_device_batch_size=per_device_batch_size,
        scorer=evaluate.load('rouge'))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_fn=eval_fn)


if __name__ == '__main__':
    fire.Fire(main)