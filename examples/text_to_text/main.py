from functools import partial
import fire
import datasets
import evaluate
import jax

from flax.core.frozen_dict import freeze

from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM

from redco import Deployer, TextToTextTrainer, get_shard_rules


def eval_rouge(eval_results, tgt_key):
    rouge_scorer = evaluate.load('rouge')

    return rouge_scorer.compute(
        predictions=[result['pred'] for result in eval_results],
        references=[result['example'][tgt_key] for result in eval_results],
        rouge_types=['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True)


def main(dataset_name='xsum',
         src_key='document',
         tgt_key='summary',
         model_name_or_path='facebook/bart-base',
         mesh_model_shards=1,
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
         jax_seed=42):
    dataset = datasets.load_dataset(dataset_name)
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with jax.default_device(jax.devices('cpu')[0]):
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, from_pt=True)
        model.params = model.to_fp32(model.params)

    deployer = Deployer(jax_seed=jax_seed, mesh_model_shards=mesh_model_shards)

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    trainer = TextToTextTrainer(
        model=model,
        params=freeze(model.params),
        optimizer=optimizer,
        learning_rate=lr_schedule_fn,
        deployer=deployer,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        src_key=src_key,
        tgt_key=tgt_key,
        params_shard_rules=get_shard_rules(model_type=model.config.model_type))

    predictor = trainer.get_default_predictor(
        gen_kwargs={'max_length': max_tgt_len, 'num_beams': num_beams})

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