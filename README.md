# Redco

Redco is a user-friendly toolkit for developing and scaling up Jax/Flax-based pipelines, 
where you can define your pipeline in a couple of functions, without concerning environmental issues, 
e.g., multi-host, parallelization for large dataset and large models, etc. 

![](https://bowentan.bitcron.com/redco_framework.jpg)

### Installation
Firstly, install [Jax](https://jax.readthedocs.io/en/latest/installation.html) based on your environment.
For example, if you are with GPUs, run
```
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Then run this command to install Redco
```
pip install -e .
```


### Examples
Examples are available in [examples/](./examples).