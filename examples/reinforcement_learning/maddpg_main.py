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

import tqdm
import fire
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo import mpe

from maddpg_agent import MADDPGAgent


def main(env_name='simple_adversary_v3',
         n_episodes=50000,
         max_steps_per_episode=25,
         learning_rate=1e-2,
         critic_loss_weight=1.,
         gamma=0.95,
         tau=0.02,
         explore_eps=0.01,
         warmup_random_steps=50000,
         replay_buffer_size=100000,
         update_interval_steps=100,
         temperature=1.,
         action_reg=1e-3,
         per_device_batch_size=1024,
         jax_seed=42):

    env = getattr(mpe, env_name).parallel_env(max_cycles=max_steps_per_episode)
    env.reset()

    state_dims = {
        agent: env.observation_space(agent=agent).shape[0]
        for agent in env.agents
    }
    action_dims = \
        {agent: env.action_space(agent=agent).n for agent in env.agents}

    maddpg = MADDPGAgent(
        agents=env.agents,
        state_dims=state_dims,
        action_dims=action_dims,
        learning_rate=learning_rate,
        critic_loss_weight=critic_loss_weight,
        replay_buffer_size=replay_buffer_size,
        warmup_random_steps=warmup_random_steps,
        update_interval_steps=update_interval_steps,
        tau=tau,
        gamma=gamma,
        temperature=temperature,
        action_reg=action_reg,
        per_device_batch_size=per_device_batch_size,
        jax_seed=jax_seed,
        workdir=f'workdir_maddpg_{env_name}')

    episode_rewards = []
    episodes = tqdm.trange(n_episodes, desc='Episodes')
    for episode_idx in episodes:
        state, _ = env.reset()
        sum_rewards = {agent: 0. for agent in env.agents}

        n_steps = 0
        while env.agents:
            explore_eps_ = explore_eps if maddpg.total_steps > warmup_random_steps else 1.
            action = {
                agent: maddpg.predict_action(
                    agent=agent,
                    agent_state=state[agent],
                    explore_eps=explore_eps_)
                for agent in env.agents
            }
            next_state, reward, done, _, _ = env.step(action)
            n_steps += 1

            sum_rewards = {
                agent: sum_rewards[agent] + reward[agent]
                for agent in sum_rewards.keys()
            }

            maddpg.add_step(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done)

            state = next_state

        episodes.set_postfix(**sum_rewards, n_steps=n_steps)
        episode_rewards.append(sum_rewards)

        if episode_idx % 1000 == 0:
            maddpg.save(episode_idx=episode_idx)

    env.close()

    for agent in episode_rewards[0].keys():
        plt.plot(
            np.arange(len(episode_rewards)),
            [r[agent] for r in episode_rewards],
            label=agent)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(env_name)
    plt.legend()
    plt.savefig(f'{maddpg.workdir}/maddpg_{env_name}.png')
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
