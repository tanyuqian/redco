from functools import partial
import fire
import numpy as np
import optax

from datasets import load_dataset
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification

from redco import Deployer, Trainer


def collate_fn(
        examples, sent0_key, sent1_key, label_key, tokenizer, max_length):
    texts = []
    for example in examples:
        if sent1_key is None:
            texts.append(example[sent0_key])
        else:
            texts.append((example[sent0_key], example[sent1_key]))

    batch = tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np')

    batch['labels'] = np.array([example[label_key] for example in examples])

    return batch


def loss_fn(train_rng, state, params, batch, is_training):
    labels = batch.pop('labels')

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, train=is_training).logits

    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()


def pred_fn(pred_rng, batch, params, model):
    logits = model(**batch, params=params, train=False).logits
    return logits.argmax(axis=-1)


def eval_metric_fn(eval_outputs):
    preds = np.array([result['pred'] for result in eval_outputs])
    labels = np.array([result['example'][1] for result in eval_outputs])
    return {'acc': np.mean(preds == labels).item()}


def main(dataset_name='sst2',
         sent0_key='sentence',
         sent1_key=None,
         label_key='label',
         model_name_or_path='roberta-large',
         max_length=512,
         n_epochs=2,
         per_device_batch_size=2,
         eval_per_device_batch_size=8,
         accumulate_grad_batches=2,
         learning_rate=4e-5,
         warmup_rate=0.1,
         weight_decay=0.,
         jax_seed=42,
         workdir='./workdir',
         run_tensorboard=False):
    dataset = load_dataset('glue', dataset_name)
    dataset = {key: list(dataset[key]) for key in dataset.keys()}

    deployer = Deployer(
        jax_seed=jax_seed, workdir=workdir, run_tensorboard=run_tensorboard)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        model_name_or_path)

    optimizer, lr_schedule_fn = deployer.get_adamw_optimizer(
        train_size=len(dataset['train']),
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        accumulate_grad_batches=accumulate_grad_batches,
        warmup_rate=warmup_rate,
        weight_decay=weight_decay)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=partial(
            collate_fn,
            sent0_key=sent0_key,
            sent1_key=sent1_key,
            label_key=label_key,
            tokenizer=tokenizer,
            max_length=max_length),
        apply_fn=model,
        loss_fn=loss_fn,
        params=model.params,
        optimizer=optimizer)

    predictor = trainer.get_default_predictor(
        pred_fn=partial(pred_fn, model=model))

    trainer.fit(
        train_examples=dataset['train'],
        per_device_batch_size=per_device_batch_size,
        n_epochs=n_epochs,
        eval_examples=dataset['validation'],
        eval_per_device_batch_size=eval_per_device_batch_size,
        eval_loss=True,
        eval_predictor=predictor,
        eval_metric_fn=eval_metric_fn)


if __name__ == '__main__':
    fire.Fire(main)
