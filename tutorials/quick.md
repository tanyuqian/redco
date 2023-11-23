### Quick Tutorial


Below is a template code to customize an arbitrary distributed training pipeline with redco.

* Redco only has three classes: `Deployer`, `Trainer`, and `Predictor`, for environmental supports, training and inference.
* No need to be a JAX expert: `numpy` is pretty enough
* No need MLSys knowledge: only specify a number `n_model_shards` to split your model
* ONLY NEED to focus on your algorithm design!

After checking out our [text classification example (glue_main.py)](examples/classification_regression/glue_main.py), you'll be an expert of redco!

```python
def collate_fn(examples, ...):
    # from raw examples to model inputs, e.g., tokenization
    return {'input_ids': input_ids, 'labels': labels}


def loss_fn(train_rng, state, params, batch, is_training, ...):
    # from model inputs defined in collate_fn, run the model and get the loss, e.g., cross_entropy
    logits = model(input_ids=batch['input_ids'], params=params)
    loss = cross_entropy(logits, batch['labels'])
    return loss


def pred_fn(pred_rng, params, batch, model, gen_kwargs):
    # from model inputs defined in collate_fn, run the model and get predictions, e.g., beam search
    batch_preds = model.generate(input_ids=batch['input_ids'],params=params)
    return batch_preds


def output_fn(batch_preds, tokenizer):
    # (optional) post process of output tensors, e.g., decode output_ids to text
    return tokenizer.batch_decode(batch_preds)


def eval_metric_fn(examples, preds):
    # (optional) given test examples and predictions, calculate evaluation metrics, e.g., Rouge-L
    return rouge_scorer.compute(
        predictions=preds,
        references=[example['target'] for example in examples],
        rouge_types=['rouge1', 'rouge2', 'rougeL'])

# define seed, workdir, tensorboard, wandb, multi-host env, etc.
deployer = redco.Deployer(
    jax_seed=jax_seed, # randomness control
    n_model_shards=n_model_shards, # how many pieces to split the model (the only number needed for model parallelism)
    workdir=workdir, run_tensorboard=True, run_wandb=True, # logging utils
    host0_address='111.222.333.444', n_processes=2 # setup multi-host env  
) 

train_examples, valid_examples = load_dataset(...) # load dataset into python-list
model, params = FlaxModel() # a model defined in flax, e.g., transformers.FlaxT5ForConditionalGeneration()
optimizer = adam(lr=0.001) # a optimizer defined in optax 

# define redco.Trainer
trainer = redco.Trainer(
    deployer=deployer,
    collate_fn=collate_fn,
    loss_fn=loss_fn, 
    params=params, 
    optimizer=optimizer,
    params_sharding_rules=deployer.get_sharding_rules(params) # automatically generated model parallelism  
)

# define redco.Predictor for prediction and evaluation during training
predictor = trainer.get_default_predictor(
    pred_fn=pred_fn, output_fn=output_fn)

# pass in your training config and run the training
trainer.fit(
    train_examples=train_examples,
    per_device_batch_size=per_device_batch_size,
    n_epochs=n_epochs,
    eval_examples=valid_examples,
    eval_per_device_batch_size=eval_per_device_batch_size,
    eval_loss=True, # if compute loss on eval_examples after each epoch
    eval_predictor=predictor, # run prediction on eval_examples after each epoch
    eval_metric_fn=eval_metric_fn, # eval_metric_fn above
    eval_sanity_check=True,
    save_every_ckpt=False,
    save_last_ckpt=True,
    save_argmin_ckpt_by_metrics=None,
    save_argmax_ckpt_by_metrics=['rouge-L'], # save the model with the best rouge-L score defined in eval_metric_fn
    save_opt_states=True)
```
