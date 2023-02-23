from collections import namedtuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from redco import Deployer, Trainer, Predictor


class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 per_device_batch_size,
                 gamma,
                 gae_lambda,
                 epsilon,
                 jax_seed=42):
        self._deployer = Deployer(jax_seed=jax_seed, verbose=False)

        self._per_device_batch_size = per_device_batch_size
        _, self._global_batch_size = self._deployer.process_batch_size(
            per_device_batch_size=per_device_batch_size)

        actor_model = MLP(output_dim=action_dim)
        self._actor_trainer, self._actor_predictor = \
            get_trainer_and_predictor(
                deployer=self._deployer,
                model=actor_model,
                learning_rate=actor_lr,
                input_dim=state_dim,
                loss_fn=partial(actor_loss_fn, epsilon=epsilon),
                pred_fn=partial(pred_fn, model=actor_model),
                output_fn=None)

        critic_model = MLP(output_dim=1)
        self._critic_trainer, self._critic_predictor = \
            get_trainer_and_predictor(
                deployer=self._deployer,
                model=critic_model,
                learning_rate=critic_lr,
                input_dim=state_dim,
                loss_fn=critic_loss_fn,
                pred_fn=partial(pred_fn, model=critic_model),
                output_fn=lambda model_output: model_output[:, 0].tolist())

        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._train_examples = []

    def predict_values(self, states):
        per_device_batch_size = 1
        while per_device_batch_size * jax.device_count() < len(states):
            per_device_batch_size *= 2

        return self._critic_predictor.predict(
            examples=[{'states': state} for state in states],
            per_device_batch_size=per_device_batch_size,
            params=self._critic_trainer.params)

    def get_actor_logits(self, states):
        per_device_batch_size = 1
        while per_device_batch_size * jax.device_count() < len(states):
            per_device_batch_size *= 2

        return self._actor_predictor.predict(
            examples=[{'states': np.asarray(state)} for state in states],
            per_device_batch_size=per_device_batch_size,
            params=self._actor_trainer.params)

    def predict_action(self, state):
        logits = jnp.array(self.get_actor_logits([state])[0])
        return jax.random.categorical(
            key=self._deployer.gen_rng(), logits=logits).item()

    def update(self, transitions):
        states = [trans.state for trans in transitions]
        next_states = [trans.next_state for trans in transitions]
        actions = jnp.array([trans.action for trans in transitions])

        v_states = self.predict_values(states=states)
        v_next_states = self.predict_values(states=next_states)

        log_probs0s = nn.log_softmax(
            jnp.array(self.get_actor_logits(states=states)))
        log_probs0s = jnp.take_along_axis(
            log_probs0s, actions[..., None], axis=-1)[..., 0]

        advantage = 0.
        for trans, v_state, v_next_state, log_probs0 in zip(
                reversed(transitions),
                reversed(v_states),
                reversed(v_next_states),
                reversed(log_probs0s)):
            td_target = trans.reward \
                        + self._gamma * v_next_state * (1. - trans.done)
            advantage = self._gamma * self._gae_lambda * advantage \
                        + (td_target - v_state)

            self._train_examples.append({
                'states': trans.state,
                'actions': trans.action,
                'td_targets': td_target,
                'advantages': advantage,
                'log_probs0': log_probs0
            })

    def train(self, n_epochs):
        per_device_batch_size = 1
        while per_device_batch_size * 2 * jax.device_count() < \
                len(self._train_examples):
            per_device_batch_size *= 2

        self._actor_trainer.fit(
            train_examples=self._train_examples,
            per_device_batch_size=per_device_batch_size,
            n_epochs=n_epochs)

        self._critic_trainer.fit(
            train_examples=self._train_examples,
            per_device_batch_size=per_device_batch_size,
            n_epochs=n_epochs)

        self._train_examples = []


Transition = namedtuple('Transition', [
    'state', 'action', 'next_state', 'reward', 'done'])


class MLP(nn.Module):
    hidden_dim: int = 128
    n_layers: int = 2
    output_dim: int = None

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        return nn.Dense(features=self.output_dim)(x)


def get_trainer_and_predictor(deployer,
                              model,
                              learning_rate,
                              input_dim,
                              loss_fn,
                              pred_fn,
                              output_fn):
    params = model.init(deployer.gen_rng(), jnp.zeros((1, input_dim)))['params']
    optimizer = optax.adam(learning_rate=learning_rate)

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=pred_fn,
        output_fn=output_fn)

    return trainer, predictor


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
