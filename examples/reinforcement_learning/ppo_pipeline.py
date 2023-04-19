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

from typing import Any

import numpy as np
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    hidden_dim: Any = 128
    n_layers: Any = 2
    output_dim: Any = None

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        return nn.Dense(features=self.output_dim)(x)


def collate_fn(examples):
    batch = {}
    for key in ['states', 'actions', 'td_targets', 'advantages', 'log_probs0']:
        if key in examples[0]:
            batch[key] = np.stack([example[key] for example in examples])

    return batch


def actor_loss_fn(train_rng, state, params, batch, is_training, epsilon):
    log_probs = nn.log_softmax(
        state.apply_fn({'params': params}, batch['states']))
    log_probs = jnp.take_along_axis(
        log_probs, batch['actions'][..., None], axis=-1)[..., 0]
    ratio = jnp.exp(log_probs - batch['log_probs0'])

    return -jnp.mean(jnp.minimum(
        ratio * batch['advantages'],
        jnp.clip(ratio, 1. - epsilon, 1. + epsilon) * batch['advantages']))


def critic_loss_fn(train_rng, state, params, batch, is_training):
    return jnp.mean(jnp.square(state.apply_fn(
        {'params': params}, batch['states'])[:, 0] - batch['td_targets']))


def pred_fn(pred_rng, batch, params, model):
    return model.apply({'params': params}, batch['states'])
