import fire
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2

from maddpg_agent import MADDPGAgent, Transition


def main(n_episodes=5000,
         learning_rate=1e-2,
         critic_loss_weight=1.,
         gamma=0.95,
         tau=1e-2,
         replay_buffer_size=100000,
         warmup_steps=4000,
         update_interval_steps=100,
         temperature=1.,
         action_reg=1e-3,
         per_device_batch_size=1024,
         jax_seed=42):

    env = simple_adversary_v2.parallel_env(
        N=2, max_cycles=25, continuous_actions=False)
    env.reset()

    maddpg = MADDPGAgent(
        agents=env.agents,
        state_dims={
            agent: env.observation_space(agent=agent).shape[0] for agent in env.agents},
        action_dims={
            agent: env.action_space(agent=agent).n for agent in env.agents},
        learning_rate=learning_rate,
        critic_loss_weight=critic_loss_weight,
        replay_buffer_size=replay_buffer_size,
        warmup_steps=warmup_steps,
        update_interval_steps=update_interval_steps,
        tau=tau,
        gamma=gamma,
        temperature=temperature,
        action_reg=action_reg,
        per_device_batch_size=per_device_batch_size,
        jax_seed=jax_seed)

    episode_rewards = []
    for episode_idx in range(n_episodes):
        state = env.reset()
        sum_rewards = {agent: 0. for agent in env.agents}

        while env.agents:
            action = {
                agent: maddpg.predict_action(
                    agent=agent, agent_state=state[agent])
                for agent in env.agents
            }
            next_state, reward, done, _, info = env.step(action)

            sum_rewards = {
                agent: sum_rewards[agent] + reward[agent]
                for agent in sum_rewards.keys()
            }

            maddpg.update(transition=Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done={key: int(value) for key, value in done.items()}))

            state = next_state

        episode_rewards.append(sum_rewards)
        print(episode_idx, sum_rewards)

    env.close()

    plt.plot(np.arange(len(episode_rewards)), episode_rewards, label='ddpg')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
