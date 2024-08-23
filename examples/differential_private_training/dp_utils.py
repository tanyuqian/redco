import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import l2_norm
from dp_accounting import dp_event
from dp_accounting import rdp


def get_noise_multiplier_by_epsilon(epsilon,
                                    steps,
                                    num_examples,
                                    batch_size,
                                    target_delta=1e-5,
                                    low=0.,
                                    high=100.,
                                    n_attempts=100):
    def _compute_epsilon(noise_multiplier):
        return compute_epsilon(
            steps=steps,
            num_examples=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta)

    for _ in range(n_attempts):
        mid = (low + high) / 2.
        if _compute_epsilon(noise_multiplier=mid) > epsilon:
            low = mid
        else:
            high = mid

    noise_multiplier = (low + high) / 2.
    return noise_multiplier, _compute_epsilon(noise_multiplier=noise_multiplier)


# Copied from
# https://github.com/google/jax/blob/main/examples/differentially_private_sgd.py
def compute_epsilon(
        steps, num_examples, batch_size, noise_multiplier, target_delta):
    if num_examples * target_delta > 1.:
        print('Your delta might be too high.')
    q = batch_size / float(num_examples)
    orders = list(jnp.linspace(1.1, 10.9, 99)) + list(range(11, 64))
    accountant = rdp.rdp_privacy_accountant.RdpAccountant(orders)
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            q, dp_event.GaussianDpEvent(noise_multiplier)), steps)
    return accountant.get_epsilon(target_delta)


# adapted from default_train_step(), added `loss_and_per_sample_grads`
# https://github.com/tanyuqian/redco/blob/master/redco/trainers/utils.py
def dp_train_step(
        rng, state, batch, loss_fn, lr_schedule_fn, mesh, compute_dtype):
    def loss_and_grads(rng_, batch_):
        return jax.value_and_grad(
            lambda params: loss_fn(
                rng=rng_,
                state=state,
                params=params,
                batch=batch_,
                is_training=True)
        )(jax.tree.map(lambda x: x.astype(compute_dtype), state.params))

    def loss_and_per_sample_grads(rng_, batch_):
        batch_ = jax.tree.map(lambda x: x[:, None], batch_)
        loss, grads = jax.vmap(lambda b: loss_and_grads(rng_, b))(batch_)

        return loss.mean(), grads

    if mesh is None:
        loss, grads = loss_and_per_sample_grads(rng, batch)
        grads = jax.lax.pmean(grads, axis_name='dp')
    else:
        loss, grads = jax.vmap(loss_and_per_sample_grads)(
            jax.random.split(rng, num=mesh.shape['dp']), batch)
        loss = jnp.mean(loss)
        grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)

    new_state = state.apply_gradients(grads=jax.tree.map(
        lambda grad, param: grad.astype(param.dtype), grads, state.params))

    metrics = {'loss': loss, 'step': state.step, 'grad_norm': l2_norm(grads)}
    if lr_schedule_fn is not None:
        metrics['lr'] = lr_schedule_fn(state.step)
    if mesh is None:
        metrics = jax.lax.pmean(metrics, axis_name='dp')

    return new_state, metrics
