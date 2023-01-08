from functools import partial
import multiprocessing

from torchmeta.datasets import helpers
from torchmeta.utils.data import CombinationRandomSampler

import numpy as np
import jax.numpy as jnp
import flax.linen as nn


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=5)(x)
        return x


def get_torchmeta_dataset(dataset_name, n_ways, n_shots, n_test_shots):
    dataset = {}
    for meta_split in ['train', 'val', 'test']:
        dataset[meta_split] = getattr(helpers, dataset_name)(
            "./data",
            ways=n_ways,
            shots=n_shots,
            test_shots=n_test_shots,
            meta_split=meta_split,
            download=True)

    return dataset


def sample_task(combination, tm_dataset):
    task = tm_dataset[combination]
    return {
        split: {
            'inputs': np.stack([np.asarray(
                inner_example[0]) for inner_example in task[split]]),
            'labels': np.stack([np.asarray(
                inner_example[1]) for inner_example in task[split]]),
        } for split in task.keys()
    }


def sample_tasks(tm_dataset, n_tasks, epoch_idx=None):
    sampler = CombinationRandomSampler(tm_dataset)
    label_combos = []
    for label_combo in sampler:
        label_combos.append(label_combo)
        if len(label_combos) == n_tasks:
            break

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        print(f'Sampling {n_tasks} tasks...')
        tasks = pool.map(
            partial(sample_task, tm_dataset=tm_dataset), label_combos)

    return tasks
