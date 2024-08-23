## Differential-Private Training

This example implements per-sample gradient noising by customizing `train_step_fn` in `redco.Trainer`.

Specifically, we provide `dp_utils.py` that includes 3 functions:
* `dp_train_step()`: Implemented per-sample gradients for DP optimizers based on the default training function inside RedCoast.
* `compute_epsilon()`: Get privacy budget `epsilon`.
* `get_noise_multiplier_by_epsilon()`: Get the corresponding `noise_multiplier` based on a given `epsilon`.

The functions in `dp_utils.py` is generally applicable to DP training algorithms (e.g., DP finetuning LLMs). 
Here we showcase the use of them on the [MNIST example](https://tanyuqian.github.io/redco/tutorial/mnist_dp/) in `main.py`.

### Requirement

```shell
pip install redco==0.4.22
```


To simulate multiple devices in cpu-only envs,
```
XLA_FLAGS="--xla_force_host_platform_device_count=8" python main.py --n_model_shards 2
```