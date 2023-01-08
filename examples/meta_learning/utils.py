from functools import partial
import multiprocessing

from torchmeta.datasets import helpers
from torchmeta.utils.data import CombinationRandomSampler

import numpy as np
import jax.numpy as jnp
import flax.linen as nn


class ConvBlock(nn.Module):
    features: int
    kernel_width: int
    pooling_width: int

    @nn.compact
    def __call__(self, x):
        return nn.Sequential([
            nn.Conv(
                features=self.features,
                kernel_size=(self.kernel_width, self.kernel_width)),
            nn.LayerNorm(),
            nn.activation.relu,
            partial(
                nn.max_pool,
                window_shape=(self.pooling_width, self.pooling_width),
                strides=(self.pooling_width, self.pooling_width))
        ])(x)


class ConvNet(nn.Module):
    conv_layers: int = 4
    features: int = 64
    kernel_width: int = 3
    pooling_width: int = 2
    classes: int = 1

    @nn.compact
    def __call__(self, x):
        for _ in range(self.conv_layers):
            x = ConvBlock(
                features=self.features,
                kernel_width=self.kernel_width,
                pooling_width=self.pooling_width
            )(x)

        return nn.Dense(features=self.classes)(x.reshape((x.shape[0], -1)))


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
                inner_example[0]) for inner_example in task[split]]
            ).transpose((0, 2, 3, 1)),
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
