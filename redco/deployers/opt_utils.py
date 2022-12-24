import optax


def get_multistep_adamw_optimizer(train_size,
                                  global_batch_size,
                                  n_epochs,
                                  learning_rate,
                                  accumulate_grad_batches,
                                  warmup_rate,
                                  weight_decay):
    total_train_steps = n_epochs * (train_size // global_batch_size)
    warmup_steps = int(total_train_steps * warmup_rate)

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate,
        transition_steps=warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0,
        transition_steps=total_train_steps - warmup_steps)
    lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule_fn, weight_decay=weight_decay),
        every_k_schedule=accumulate_grad_batches)

    return optimizer
