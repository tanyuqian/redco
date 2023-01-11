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
                 lambda_td,
                 epsilon,
                 jax_seed=42):
        self._deployer = Deployer(jax_seed=jax_seed)

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
                output_fn=lambda x: x.tolist())

        critic_model = MLP(output_dim=1)
        self._critic_trainer, self._critic_predictor = \
            get_trainer_and_predictor(
                deployer=self._deployer,
                model=critic_model,
                learning_rate=critic_lr,
                input_dim=state_dim,
                loss_fn=critic_loss_fn,
                pred_fn=partial(pred_fn, model=critic_model),
                output_fn=lambda x: x[:, 0].tolist())

        self._gamma = gamma
        self._lambda_td = lambda_td
        self._train_examples = []

    def predict_values(self, states):
        batch_size = 1
        while batch_size < len(states):
            batch_size *= 2

        return self._critic_predictor.predict(
            examples=[{'states': state} for state in states],
            per_device_batch_size=batch_size,
            params=self._critic_trainer.params)

    def get_actor_logits(self, states):
        batch_size = 1
        while batch_size < len(states):
            batch_size *= 2

        return self._actor_predictor.predict(
            examples=[{'states': np.asarray(state)} for state in states],
            per_device_batch_size=batch_size,
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
        for t in reversed(list(range(0, len(transitions)))):
            td_target = \
                transitions[t].reward \
                + self._gamma * v_next_states[t] * (1. - transitions[t].done)
            advantage = \
                self._gamma * self._lambda_td * advantage \
                + (td_target - v_states[t])

            self._train_examples.append({
                'states': transitions[t].state,
                'actions': transitions[t].action,
                'td_targets': td_target,
                'advantages': advantage,
                'log_probs0': log_probs0s[t]
            })

    def train(self, n_epochs):
        batch_size = 1
        while batch_size * 2 <= len(self._train_examples):
            batch_size *= 2

        self._actor_trainer.fit(
            train_examples=self._train_examples,
            per_device_batch_size=batch_size,
            n_epochs=n_epochs)

        self._critic_trainer.fit(
            train_examples=self._train_examples,
            per_device_batch_size=batch_size,
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


def get_trainer_and_predictor(
        deployer, model, learning_rate, input_dim, loss_fn, pred_fn, output_fn):
    params = model.init(deployer.gen_rng(), jnp.zeros((1, input_dim)))['params']
    optimizer = optax.adam(learning_rate=learning_rate)
    dummy_example = {
        'states': np.zeros(input_dim),
        'actions': 0,
        'td_targets': 0.,
        'advantages': 0.,
        'log_probs0': 0.
    }

    trainer = Trainer(
        deployer=deployer,
        collate_fn=collate_fn,
        apply_fn=model.apply,
        loss_fn=loss_fn,
        params=params,
        optimizer=optimizer,
        lr_schedule_fn=lambda t: learning_rate,
        dummy_example=dummy_example)

    predictor = Predictor(
        deployer=deployer,
        collate_fn=collate_fn,
        pred_fn=pred_fn,
        output_fn=output_fn,
        dummy_example=dummy_example)

    return trainer, predictor


def collate_fn(examples):
    batch = {}
    for key in ['states', 'actions', 'td_targets', 'advantages', 'log_probs0']:
        if key in examples[0]:
            batch[key] = np.stack([example[key] for example in examples])

    return batch


def actor_loss_fn(state, params, batch, train, epsilon):
    log_probs = nn.log_softmax(
        state.apply_fn({'params': params}, batch['states']))
    log_probs = jnp.take_along_axis(
        log_probs, batch['actions'][..., None], axis=-1)[..., 0]
    ratio = jnp.exp(log_probs - batch['log_probs0'])

    return -jnp.mean(jnp.minimum(
        ratio * batch['advantages'],
        jnp.clip(ratio, 1. - epsilon, 1. + epsilon) * batch['advantages']))


def critic_loss_fn(state, params, batch, train):
    return jnp.mean(jnp.square(state.apply_fn(
        {'params': params}, batch['states'])[:, 0] - batch['td_targets']))


def pred_fn(batch, params, model):
    return model.apply({'params': params}, batch['states'])
