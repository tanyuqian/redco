from collections import deque, namedtuple
from functools import partial
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax

from redco import Deployer, Trainer, Predictor
from ddpg_pipeline import (
    Actor,
    Critic,
    collate_fn,
    loss_fn,
    actor_pred_fn,
    critic_pred_fn)


Transition = namedtuple('Transition', [
    'state', 'action', 'next_state', 'reward', 'done'])


class DDPGAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 learning_rate,
                 critic_loss_weight,
                 replay_buffer_size,
                 warmup_steps,
                 sigma,
                 tau,
                 gamma,
                 per_device_batch_size,
                 jax_seed):
        self._deployer = Deployer(jax_seed=jax_seed, verbose=False)

        critic = Critic()
        actor = Actor(action_dim=action_dim, action_bound=action_bound)

        self._target_critic_params = critic.init(
            self._deployer.gen_rng(),
            states=jnp.zeros((1, state_dim)),
            actions=jnp.zeros((1, action_dim))
        )['params']

        self._target_actor_params = actor.init(
            self._deployer.gen_rng(), states=jnp.zeros((1, state_dim))
        )['params']

        self._trainer = Trainer(
            deployer=self._deployer,
            collate_fn=collate_fn,
            apply_fn=lambda: None,
            loss_fn=partial(
                loss_fn,
                actor=actor,
                critic=critic,
                critic_loss_weight=critic_loss_weight),
            params={
                'actor': self._target_actor_params,
                'critic': self._target_critic_params},
            optimizer=optax.adam(learning_rate=learning_rate))

        self._critic_predictor = Predictor(
            deployer=self._deployer,
            collate_fn=collate_fn,
            pred_fn=partial(critic_pred_fn, critic=critic))

        self._actor_predictor = Predictor(
            deployer=self._deployer,
            collate_fn=collate_fn,
            pred_fn=partial(actor_pred_fn, actor=actor))

        self._replay_buffer = deque(maxlen=replay_buffer_size)
        self._warmup_steps = warmup_steps
        self._sigma = sigma
        self._gamma = gamma
        self._tau = tau
        _, self._global_batch_size = self._deployer.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        self._per_device_batch_size = per_device_batch_size

    def predict_action(self, state):
        action = self._actor_predictor.predict(
            examples=[{'states': state}],
            params=self._trainer.params['actor'],
            per_device_batch_size=1)[0]

        return action + self._sigma * np.random.randn(*action.shape)

    def get_target_actions(self, states):
        return self._actor_predictor.predict(
            examples=[{'states': state} for state in states],
            params=self._target_actor_params,
            per_device_batch_size=self._per_device_batch_size)

    def get_target_q_values(self, states, actions):
        return self._critic_predictor.predict(
            examples=[
                {'states': state, 'actions': action}
                for state, action in zip(states, actions)],
            params=self._target_critic_params,
            per_device_batch_size=self._per_device_batch_size)

    def update(self, transition):
        self._replay_buffer.append(transition)

        if len(self._replay_buffer) > self._warmup_steps:
            batch = random.sample(self._replay_buffer, self._global_batch_size)
            self.train(batch)

    def update_target(self):
        self._target_actor_params = jax.tree_util.tree_map(
            lambda x, y: (1. - self._tau) * x + self._tau * y,
            self._target_actor_params,
            self._trainer.params['actor'])

        self._target_critic_params = jax.tree_util.tree_map(
            lambda x, y: (1. - self._tau) * x + self._tau * y,
            self._target_critic_params,
            self._trainer.params['critic'])

    def train(self, transitions):
        states = [trans.state for trans in transitions]
        next_states = [trans.next_state for trans in transitions]
        actions = [trans.action for trans in transitions]
        rewards = np.array([trans.reward for trans in transitions])
        dones = np.array([trans.done for trans in transitions])

        next_actions = self.get_target_actions(states=next_states)
        next_q_values = self.get_target_q_values(
            states=next_states, actions=next_actions)
        td_targets = \
            rewards + self._gamma * np.array(next_q_values) * (1. - dones)

        examples = [
            {'states': state, 'actions': action, 'td_targets': td_target}
            for state, action, td_target in zip(states, actions, td_targets)]
        self._trainer.train(
            examples=examples,
            per_device_batch_size=self._per_device_batch_size)

        self.update_target()