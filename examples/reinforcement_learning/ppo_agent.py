from collections import namedtuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from redco import Deployer, Trainer, Predictor
from ppo_pipeline import MLP, collate_fn, actor_loss_fn, critic_loss_fn, pred_fn


Transition = namedtuple('Transition', [
    'state', 'action', 'next_state', 'reward', 'done'])


class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 gamma,
                 gae_lambda,
                 epsilon,
                 jax_seed=42):
        self._deployer = Deployer(jax_seed=jax_seed, verbose=False)

        self._actor_trainer, self._actor_predictor = \
            self.get_trainer_and_predictor(
                model=MLP(output_dim=action_dim),
                learning_rate=actor_lr,
                input_dim=state_dim,
                loss_fn=partial(actor_loss_fn, epsilon=epsilon),
                output_fn=None)

        self._critic_trainer, self._critic_predictor = \
            self.get_trainer_and_predictor(
                model=MLP(output_dim=1),
                learning_rate=critic_lr,
                input_dim=state_dim,
                loss_fn=critic_loss_fn,
                output_fn=lambda model_output: model_output[:, 0].tolist())

        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._train_examples = []

    def get_trainer_and_predictor(self,
                                  model,
                                  input_dim,
                                  loss_fn,
                                  output_fn,
                                  learning_rate):
        params = model.init(
            self._deployer.gen_rng(), jnp.zeros((1, input_dim)))['params']
        optimizer = optax.adam(learning_rate=learning_rate)

        trainer = Trainer(
            deployer=self._deployer,
            collate_fn=collate_fn,
            apply_fn=model.apply,
            loss_fn=loss_fn,
            params=params,
            optimizer=optimizer)

        predictor = Predictor(
            deployer=self._deployer,
            collate_fn=collate_fn,
            pred_fn=partial(pred_fn, model=model),
            output_fn=output_fn)

        return trainer, predictor

    def get_per_device_batch_size(self, n_examples):
        per_device_batch_size = 1
        while True:
            _, global_batch_size = self._deployer.process_batch_size(
                per_device_batch_size=per_device_batch_size)
            if global_batch_size * 2 <= n_examples:
                per_device_batch_size *= 2
            else:
                break

        return per_device_batch_size

    def predict_values(self, states):
        per_device_batch_size = self.get_per_device_batch_size(
            n_examples=len(states))

        return self._critic_predictor.predict(
            examples=[{'states': state} for state in states],
            per_device_batch_size=per_device_batch_size,
            params=self._critic_trainer.params)

    def get_actor_logits(self, states):
        per_device_batch_size = self.get_per_device_batch_size(
            n_examples=len(states))

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
        per_device_batch_size = self.get_per_device_batch_size(
            n_examples=len(self._train_examples))

        self._actor_trainer.fit(
            train_examples=self._train_examples,
            per_device_batch_size=per_device_batch_size,
            n_epochs=n_epochs)

        self._critic_trainer.fit(
            train_examples=self._train_examples,
            per_device_batch_size=per_device_batch_size,
            n_epochs=n_epochs)

        self._train_examples = []
