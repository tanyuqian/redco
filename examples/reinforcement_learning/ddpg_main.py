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

import fire
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from ddpg_agent import DDPGAgent, Transition


def main(env_name='Pendulum-v1',
         n_episodes=200,
         learning_rate=3e-4,
         critic_loss_weight=10.,
         gamma=0.98,
         tau=0.005,
         replay_buffer_size=10000,
         warmup_steps=1000,
         per_device_batch_size=64,
         sigma=0.01,
         jax_seed=42):
    env = gym.make(env_name)

    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bound=env.action_space.high[0],
        learning_rate=learning_rate,
        critic_loss_weight=critic_loss_weight,
        replay_buffer_size=replay_buffer_size,
        warmup_steps=warmup_steps,
        sigma=sigma,
        tau=tau,
        gamma=gamma,
        per_device_batch_size=per_device_batch_size,
        jax_seed=jax_seed)

    episode_rewards = []
    for episode_idx in range(n_episodes):
        sum_rewards = 0.
        state, info = env.reset()
        while True:
            action = agent.predict_action(state=state)
            next_state, reward, terminated, truncated, info = env.step(action)

            sum_rewards += reward
            agent.update(transition=Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=int(terminated)))

            state = next_state

            if terminated or truncated:
                print(f'Episode {episode_idx}: reward = {sum_rewards}')
                episode_rewards.append(sum_rewards)
                break

    env.close()

    plt.plot(np.arange(len(episode_rewards)), episode_rewards, label='ddpg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(env_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
