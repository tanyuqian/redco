## Differential-Private Training

A MNIST example of data-private training. It implements per-sample gradient noising by modifying `default_train_step` and pass into `redco.Trainer`.

### Requirement

```shell
pip install redco==0.4.22
```


To simulate multiple devices in cpu-only envs,
```
XLA_FLAGS="--xla_force_host_platform_device_count=8" python main.py --n_model_shards 2
```