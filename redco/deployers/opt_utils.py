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


def get_lr_schedule_fn(schedule_type,
                       total_train_steps,
                       warmup_steps,
                       init_learning_rate,
                       learning_rate,
                       end_learning_rate):
    warmup_fn = optax.linear_schedule(
        init_value=init_learning_rate,
        end_value=learning_rate,
        transition_steps=warmup_steps)

    if schedule_type == 'linear':
        decay_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=end_learning_rate,
            transition_steps=total_train_steps - warmup_steps)
    elif schedule_type == 'cosine':
        decay_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_train_steps - warmup_steps,
            alpha=end_learning_rate / learning_rate)
    else:
        raise ValueError(f'lr schedule_type={schedule_type} not supported now.')

    lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps])

    return lr_schedule_fn
