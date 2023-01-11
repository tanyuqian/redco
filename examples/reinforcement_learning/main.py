import fire
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from ppo_utils import PPOAgent, Transition


def main(env_name='CartPole-v1',
         actor_lr=1e-3,
         critic_lr=1e-2,
         gamma=0.98,
         lambda_td=0.95,
         epsilon=0.2,
         jax_seed=42):
    env = gym.make(env_name)
    assert len(env.observation_space.shape) == 1

    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        per_device_batch_size=32,
        gamma=gamma,
        lambda_td=lambda_td,
        epsilon=epsilon,
        jax_seed=jax_seed)

    episode_rewards = []
    for episode_idx in range(1000):
        sum_rewards = 0.
        state, info = env.reset()
        transitions = []
        while True:
            action = agent.predict_action(state=state)
            next_state, reward, terminated, truncated, info = env.step(action)

            sum_rewards += reward
            transitions.append(Transition(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=int(terminated)))

            state = next_state

            if terminated or truncated:
                print(f'Episode {episode_idx}: reward = {sum_rewards}')
                episode_rewards.append(sum_rewards)
                agent.update(transitions=transitions)
                agent.train(n_epochs=10)
                break

    env.close()

    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
