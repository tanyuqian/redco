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

from collections import deque, namedtuple
from functools import partial
import random
import numpy as np
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze
import optax
from scipy.special import softmax

from redco import Deployer, Trainer, Predictor
from maddpg_pipeline import (
    Actor,
    Critic,
    collate_fn,
    loss_fn,
    actor_pred_fn,
    critic_pred_fn)


Transition = namedtuple('Transition', [
    'state', 'action', 'next_state', 'reward', 'done'])


class MADDPGAgent:
    def __init__(self,
                 agents,
                 state_dims,
                 action_dims,
                 learning_rate=1e-2,
                 critic_loss_weight=1.,
                 replay_buffer_size=100000,
                 minimal_buffer_size=4000,
                 update_interval_steps=100,
                 tau=0.02,
                 gamma=0.95,
                 temperature=1.,
                 action_reg=1e-3,
                 per_device_batch_size=1024,
                 jax_seed=42,
                 workdir=None,
                 init_params_path=None):
        self._deployer = Deployer(
            jax_seed=jax_seed, verbose=False, workdir=workdir)

        self._agents = agents
        self._state_dims = state_dims
        self._action_dims = action_dims

        self._target_critic_params = {}
        self._target_actor_params = {}
        self._trainer = {}
        self._critic_predictor = {}
        self._actor_predictor = {}
        self._replay_buffer = {}

        if init_params_path is not None:
            init_params = self._deployer.load_params(filepath=init_params_path)
            for agent in self._agents:
                self._target_critic_params[agent] = init_params[agent]['critic']
                self._target_actor_params[agent] = init_params[agent]['actor']

        for agent_idx, agent in enumerate(agents):
            critic = Critic()
            actor = Actor(action_dim=self._action_dims[agent])

            if init_params_path is None:
                self._target_critic_params[agent] = critic.init(
                    self._deployer.gen_rng(),
                    states=jnp.zeros((1, sum(self._state_dims.values()))),
                    actions=jnp.zeros((1, sum(self._action_dims.values())))
                )['params']

                self._target_actor_params[agent] = actor.init(
                    self._deployer.gen_rng(),
                    states=jnp.zeros((1, self._state_dims[agent]))
                )['params']

            agent_action_idx_l = sum(
                [self._action_dims[agents[i]] for i in range(agent_idx)])
            agent_action_idx_r = sum(
                [self._action_dims[agents[i]]
                 for i in range(agent_idx + 1)])

            self._trainer[agent] = Trainer(
                deployer=self._deployer,
                collate_fn=collate_fn,
                apply_fn=lambda: None,
                loss_fn=partial(
                    loss_fn,
                    actor=actor,
                    critic=critic,
                    critic_loss_weight=critic_loss_weight,
                    temperature=temperature,
                    action_reg=action_reg,
                    agent_action_idx_l=agent_action_idx_l,
                    agent_action_idx_r=agent_action_idx_r),
                params={
                    'actor': self._target_actor_params[agent],
                    'critic': self._target_critic_params[agent]},
                optimizer=optax.adam(learning_rate=learning_rate))

            self._critic_predictor[agent] = Predictor(
                deployer=self._deployer,
                collate_fn=collate_fn,
                pred_fn=partial(critic_pred_fn, critic=critic))

            self._actor_predictor[agent] = Predictor(
                deployer=self._deployer,
                collate_fn=collate_fn,
                pred_fn=partial(actor_pred_fn, actor=actor))

        self._replay_buffer = deque(maxlen=replay_buffer_size)

        self._update_interval_steps = update_interval_steps
        self._minimal_buffer_size = minimal_buffer_size
        self._gamma = gamma
        self._tau = tau
        _, self._global_batch_size = self._deployer.process_batch_size(
            per_device_batch_size=per_device_batch_size)
        self._per_device_batch_size = per_device_batch_size
        self._total_steps = 0

    def predict_action(self, agent, agent_state, explore_eps):
        if random.random() < explore_eps:
            return random.randint(0, self._action_dims[agent] - 1)

        action_logits = self._actor_predictor[agent].predict(
            examples=[{'states': agent_state}],
            params=self._trainer[agent].params['actor'],
            per_device_batch_size=1)[0]

        return np.random.choice(
            self._action_dims[agent], p=softmax(action_logits))

    def get_target_actions(self, states):
        actions = [{} for _ in range(len(states))]
        for agent in self._agents:
            action_logits = self._actor_predictor[agent].predict(
                examples=[{'states': state[agent]} for state in states],
                params=self._target_actor_params[agent],
                per_device_batch_size=self._per_device_batch_size)

            for i in range(len(states)):
                actions[i][agent] = np.random.choice(
                    self._action_dims[agent], p=softmax(action_logits[i]))

        return actions

    def get_target_q_values(self, agent, states, actions):
        examples = [{
            'states': self.get_state_input(state),
            'actions': self.get_action_input(action)
        } for state, action in zip(states, actions)]

        return self._critic_predictor[agent].predict(
            examples=examples,
            params=self._target_critic_params[agent],
            per_device_batch_size=self._per_device_batch_size)

    def add_step(self, state, action, reward, next_state, done):
        self._replay_buffer.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done={agent: int(value) for agent, value in done.items()}))

        self._total_steps += 1

        if len(self._replay_buffer) >= self._minimal_buffer_size and \
                self._total_steps % self._update_interval_steps == 0:
            self.train()

    def update_target(self, agent):
        self._target_actor_params[agent] = jax.tree_util.tree_map(
            lambda x, y: (1. - self._tau) * x + self._tau * y,
            self._target_actor_params[agent],
            unfreeze(self._trainer[agent].params['actor']))

        self._target_critic_params[agent] = jax.tree_util.tree_map(
            lambda x, y: (1. - self._tau) * x + self._tau * y,
            self._target_critic_params[agent],
            unfreeze(self._trainer[agent].params['critic']))

    def train(self):
        for agent in self._agents:
            transitions = random.sample(
                self._replay_buffer, self._global_batch_size)

            next_states = [trans.next_state for trans in transitions]

            next_actions = self.get_target_actions(states=next_states)
            next_q_values = np.array(self.get_target_q_values(
                agent=agent, states=next_states, actions=next_actions))

            agent_rewards = np.array(
                [trans.reward[agent] for trans in transitions])
            agent_dones = np.array(
                [trans.done[agent] for trans in transitions])

            td_targets = (agent_rewards +
                          self._gamma * next_q_values * (1. - agent_dones))

            examples = [{
                'states': self.get_state_input(trans.state),
                'actions': self.get_action_input(trans.action),
                'td_targets': td_target,
                'actor_states': trans.state[agent]
            } for trans, td_target in zip(transitions, td_targets)]

            self._trainer[agent].train(
                examples=examples,
                per_device_batch_size=self._per_device_batch_size)

            self.update_target(agent=agent)

    def get_state_input(self, state):
        x = [state[agent] for agent in self._agents]
        return np.concatenate(x, axis=-1)

    def get_action_input(self, action):
        x = [
            np.eye(self._action_dims[agent])[action[agent]]
            for agent in self._agents
        ]
        return np.concatenate(x, axis=-1)

    def save(self, episode_idx):
        params = {agent: self._trainer[agent].params for agent in self._agents}
        save_dir = self._deployer.workdir

        self._deployer.save_params(
            params=params,
            ckpt_dir=save_dir,
            desc=f'maddpg_episode{episode_idx}')
        print(f'Checkpoint \"maddpg_episode{episode_idx}\" saved.')

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def workdir(self):
        return self._deployer.workdir
