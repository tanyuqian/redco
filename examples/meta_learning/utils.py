import os
import tqdm

from torchmeta.datasets import helpers
from torchmeta.utils.data import BatchMetaDataLoader

import numpy as np
import jax
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


def get_few_shot_dataset(dataset_name,
                         n_ways,
                         n_shots,
                         n_test_shots,
                         n_meta_tasks):
    dataset = {}
    for meta_split in ['train', 'val', 'test']:
        raw_dataset = getattr(helpers, dataset_name)(
            "./data",
            ways=n_ways,
            shots=n_shots,
            test_shots=n_test_shots,
            meta_split=meta_split,
            download=True)

        task_loader = BatchMetaDataLoader(raw_dataset, batch_size=1)

        dataset[meta_split] = []
        for task in tqdm.tqdm(task_loader,
                              total=n_meta_tasks,
                              desc=f'Loading {meta_split} data'):
            task = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], task)
            dataset[meta_split].append({
                split: {'inputs': task[split][0], 'labels': task[split][1]}
                for split in task.keys()
            })

            if len(dataset[meta_split]) == n_meta_tasks:
                print(meta_split, len(dataset[meta_split]))
                break

    return dataset
