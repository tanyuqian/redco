## MNIST

A trivial MNIST example with RedCoast.

To simulate multiple devices in cpu-only envs,
```
XLA_FLAGS="--xla_force_host_platform_device_count=8" python main.py --n_model_shards 2
```