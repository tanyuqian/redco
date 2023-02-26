import numpy as np
import jax.numpy as jnp
import optax


def collate_fn(examples, tokenizer, text_key, max_length):
    batch = tokenizer(
        [(example[text_key] + tokenizer.eos_token) for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False,
        return_tensors='np')

    batch['labels'] = np.copy(batch['input_ids'])
    batch['input_ids'][:, 1:] = batch['input_ids'][:, :-1]
    batch['input_ids'][:, 0] = tokenizer.eos_token_id

    return batch


def loss_fn(train_rng, state, params, batch, is_training, model_type):
    labels = batch.pop("labels")
    label_weights = batch['attention_mask']

    if model_type != 'opt':
        is_training_kwarg = {'train': is_training}
    else:
        is_training_kwarg = {'deterministic': not is_training}

    logits = state.apply_fn(
        **batch, params=params, dropout_rng=train_rng, **is_training_kwarg)[0]

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels)

    return jnp.sum(loss * label_weights) / jnp.sum(label_weights)


def pred_fn(pred_rng, batch, params, model, gen_kwargs):
    output_ids = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        params=params,
        prng_key=pred_rng,
        **gen_kwargs)
    return output_ids.sequences


def output_fn(batch_preds, tokenizer):
    return tokenizer.batch_decode(batch_preds, skip_special_tokens=True)
