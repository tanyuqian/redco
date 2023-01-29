from functools import partial
import fire
import datasets
import evaluate
import jax

from transformers import \
    AutoTokenizer, FlaxAutoModelForSeq2SeqLM, GenerationConfig

from redco import Deployer, TextToTextTrainer


def eval_rouge(eval_outputs, tgt_key):
    rouge_scorer = evaluate.load('rouge')

    return rouge_scorer.compute(
        predictions=[result['pred'] for result in eval_outputs],
        references=[result['example'][tgt_key] for result in eval_outputs],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(dataset_name='xsum',
         src_key='document',
         tgt_key='summary',
         model_name_or_path='facebook/bart-base',
         n_model_shards=1,
         n_epochs=2,
         per_device_batch_size=8,
         eval_per_device_batch_size=16,
         accumulate_grad_batches=2,
         max_src_len=512,
         max_tgt_len=64,
         num_beams=4,
         learning_rate=4e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         jax_seed=42,
         workdir='./workdir',
         run_tensorboard=False):
    dataset = datasets.load_dataset(dataset_name)
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with jax.default_device(jax.devices('cpu')[0]):
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, from_pt=True)
        model.params = model.to_fp32(model.params)

    try:
        generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    except:
        generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.update(max_length=max_tgt_len, num_beams=num_beams)

    deployer = Deployer(
        jax_seed=jax_seed,
        n_model_shards=n_model_shards,
        workdir=workdir,
        run_tensorboard=run_tensorboard,
        verbose=True)

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    trainer = TextToTextTrainer(
        deployer=deployer,
        model=model,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        src_key=src_key,
        tgt_key=tgt_key,
        params_shard_rules=deployer.guess_shard_rules(params=model.params))

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
        eval_metric_fn=partial(eval_rouge, tgt_key=tgt_key))


if __name__ == '__main__':
    fire.Fire(main)