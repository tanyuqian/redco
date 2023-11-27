## Faderated Learning (FedAvg)

This example implements FedAvg with Redco. 
It supports assigning a dataset from [torchvision](https://pytorch.org/vision/stable/index.html).

### Requirement

Install TorchVision
```shell
pip install torchvision==0.16.1
```

Install Redco
```shell
pip install redco==0.4.13
```

### Usage

```shell
python main.py \
    --dataset_name MNIST \
    --n_clients 100 \
    --n_data_shards 200 \
    --n_rounds 100 \
    --n_clients_per_round 10 \
    --n_client_epochs_per_round 5 \
    --per_device_batch_size 64 \
    --do_iid_partition
```
See `def main(...)` in [glue_main.py](glue_main.py) for all the tunable arguments. 

### Results

| Setting           | Command                           | Accuracy |
|-------------------|-----------------------------------|----------|
| IID Partition     | `python main.py --do_iid_partition` | 98.82%   |
| Non-IID Partition | `python main.py`                    | 97.53%   |