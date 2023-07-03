#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import optax


def get_multistep_adamw_optimizer(train_size,
                                  global_batch_size,
                                  n_epochs,
                                  learning_rate,
                                  weight_decay,
                                  warmup_rate,
                                  warmup_steps,
                                  b1,
                                  b2,
                                  eps,
                                  lr_schedule_type,
                                  end_learning_rate,
                                  accumulate_grad_batches):
    total_train_steps = n_epochs * (train_size // global_batch_size)

    if warmup_steps is None:
        warmup_steps = int(total_train_steps * warmup_rate)

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate,
        transition_steps=warmup_steps)

    if lr_schedule_type == 'linear':
        decay_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=end_learning_rate,
            transition_steps=total_train_steps - warmup_steps)
    elif lr_schedule_type == 'cosine':
        decay_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_train_steps - warmup_steps,
            alpha=end_learning_rate / learning_rate)
    else:
        raise ValueError(f'lr schedule {lr_schedule_type} not supported now.')

    lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])

    if accumulate_grad_batches == 1:
        optimizer = optax.adamw(
            learning_rate=lr_schedule_fn,
            weight_decay=weight_decay,
            b1=b1,
            b2=b2,
            eps=eps)
    else:
        optimizer = optax.MultiSteps(optax.adamw(
            learning_rate=lr_schedule_fn,
            weight_decay=weight_decay,
            b1=b1,
            b2=b2,
            eps=eps
        ), every_k_schedule=accumulate_grad_batches)

    return optimizer, lr_schedule_fn
