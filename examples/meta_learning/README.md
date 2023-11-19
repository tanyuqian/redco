## Meta Learning (MAML)

### Requirement

Install `learn2learn` for few-shot datasets.
```shell
pip install learn2learn
```

Install Redco
```shell
pip install redco==0.4.11
```

### Use

```shell
python main.py \
    --dataset_name omniglot \
    --n_ways 5 \
    --n_shots 1 \
    --n_tasks_per_epoch 1024 \
    --n_epochs 1000 \
    --per_device_batch_size 32 \
    --learning_rate 0.003 \
    --inner_learning_rate 0.5 \
    --inner_n_steps
```
With the default hyperparameters (`python main.py`), and a simple CNN modeling copied from [Flax MNIST example](https://github.com/google/flax/blob/main/examples/mnist/train.py#L36), this code gets an accuracy of `98%` on `omniglot (5-way, 1-shot)` .