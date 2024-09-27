import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict


def get_lora(model_params, lora_rank, rng):
    initializer = jax.nn.initializers.he_uniform()
    lora_params_flat = {}
    for path, param in flatten_dict(model_params).items():
        if path[-1] == 'kernel' and len(param.shape) == 2:
            lora_params_flat[path[:-1]] = {
                'a': initializer(rng, (param.shape[0], lora_rank), jnp.float32),
                'b': jnp.zeros((lora_rank, param.shape[1]))
            }

    return unflatten_dict(lora_params_flat)


def aggregate_params(model_params, lora_params, lora_alpha):
    model_params_flat = flatten_dict(model_params)
    lora_params_flat = flatten_dict(lora_params)

    params_flat = {}
    for path, param in model_params_flat.items():
        if path[-1] == 'kernel' and len(param.shape) == 2:
            a = lora_params_flat[(*path[:-1], 'a')]
            b = lora_params_flat[(*path[:-1], 'b')]
            params_flat[path] = param + jnp.matmul(a, b) * lora_alpha
        else:
            params_flat[path] = param

    return unflatten_dict(params_flat)
