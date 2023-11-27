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

import numpy as np
from torchvision import datasets


def get_dataset(data_dir, dataset_name, n_clients, n_data_shards):
    dataset = {
        'train': list(getattr(datasets, dataset_name)(
            data_dir, train=True, download=True)),
        'test': list(getattr(datasets, dataset_name)(
            data_dir, train=False, download=True))
    }

    data_class = getattr(datasets, dataset_name)
    client_data_idxes = non_iid_partition(
        dataset=data_class(data_dir, train=True, download=True),
        clients=n_clients,
        total_shards=n_data_shards,
        shards_size=len(dataset['train']) // n_data_shards,
        num_shards_per_client=n_data_shards // n_clients)

    client_train_datasets = {}
    for key in client_data_idxes:
        client_train_datasets[key] = []
        for idx in client_data_idxes[key]:
            image = np.asarray(dataset['train'][idx][0])
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            client_train_datasets[key].append({
                'image': image, 'label': dataset['train'][idx][1]
            })

    test_dataset = []
    for image, label in dataset['test']:
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        test_dataset.append({'image': image, 'label': label})

    return client_train_datasets, test_dataset


# copied from https://github.com/ayushm-agrawal/
# Federated-Learning-Implementations/blob/master/FederatedAveraging.ipynb
def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients

    params:
      - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
      - clients (int): Number of Clients to split the data between

    returns:
      - Dictionary of image indexes for each client
    """

    num_items_per_client = int(len(dataset) / clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]

    for i in range(clients):
        client_dict[i] = set(
            np.random.choice(image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])

    return client_dict


# copied from https://github.com/ayushm-agrawal/
# Federated-Learning-Implementations/blob/master/FederatedAveraging.ipynb
def non_iid_partition(dataset, clients, total_shards, shards_size,
                      num_shards_per_client):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the digit label
    Divide the data into N shards of size S
    Each of the clients will get X shards

    params:
      - dataset (torch.utils.Dataset): Dataset containing the MNIST Images
      - clients (int): Number of Clients to split the data between
      - total_shards (int): Number of shards to partition the data in
      - shards_size (int): Size of each shard
      - num_shards_per_client (int): Number of shards of size shards_size that
      each client receives

    returns:
      - Dictionary of image indexes for each client
    """

    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}
    idxs = np.arange(len(dataset))
    data_labels = np.asarray(dataset.targets)

    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(clients):
        rand_set = set(
            np.random.choice(shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate((
                client_dict[i],
                idxs[rand * shards_size:(rand + 1) * shards_size]
            ), axis=0)

    return client_dict
