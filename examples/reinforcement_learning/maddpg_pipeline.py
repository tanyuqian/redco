import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class Actor(nn.Module):
    hidden_dim: int = 64
    n_layers: int = 3
    action_dim: int = 1

    @nn.compact
    def __call__(self, states):
        x = states

        for _ in range(self.n_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        return nn.Dense(features=self.action_dim)(x)


class Critic(nn.Module):
    hidden_dim: int = 64
    action_dim: int = 1
    n_layers: int = 3

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)

        for _ in range(self.n_layers - 1):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)

        return nn.Dense(features=1)(x)


def collate_fn(examples):
    batch = {}
    for key in ['states', 'actions', 'td_targets', 'actor_states']:
        if key in examples[0]:
            batch[key] = np.stack([example[key] for example in examples])

    return batch


def gumbel_softmax(rng, logits, temperature):
    logits = logits + jax.random.gumbel(key=rng, shape=logits.shape)
    y = jax.nn.softmax(logits / temperature)
    y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), num_classes=y.shape[-1])

    return y_hard - jax.lax.stop_gradient(y) + y


def loss_fn(train_rng,
            state,
            params,
            batch,
            is_training,
            actor,
            critic,
            critic_loss_weight,
            temperature,
            action_reg,
            agent_action_idx_l,
            agent_action_idx_r):
    critic_loss = jnp.mean(jnp.square(critic.apply(
        {'params': params['critic']},
        states=batch['states'],
        actions=batch['actions']
    )[:, 0] - batch['td_targets']))

    action_logits = actor.apply(
        {'params': params['actor']}, batch['actor_states'])

    actions = jnp.concatenate([
        batch['actions'][:, :agent_action_idx_l],
        gumbel_softmax(
            rng=train_rng, logits=action_logits, temperature=temperature),
        batch['actions'][:, agent_action_idx_r:]
    ], axis=-1)

    # actions[:, agent_action_idx_l:agent_action_idx_r] = gumbel_softmax(
    #     rng=train_rng, logits=action_logits, temperature=temperature)

    q_values = critic.apply(
        {'params': jax.lax.stop_gradient(params['critic'])},
        states=batch['states'],
        actions=actions)

    actor_loss = \
        -jnp.mean(q_values) + jnp.mean(jnp.square(action_logits)) * action_reg

    return critic_loss * critic_loss_weight + actor_loss


def actor_pred_fn(pred_rng, batch, params, actor):
    return actor.apply({'params': params}, batch['states'])


def critic_pred_fn(pred_rng, batch, params, critic):
    return critic.apply(
        {'params': params}, states=batch['states'], actions=batch['actions']
    )[:, 0]
