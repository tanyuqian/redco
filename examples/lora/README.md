## Sequence-to-Sequence

This example implements LoRA on a seq2seq task, by default with `bart-base` as the backbone.

Following the data configuration in [this HuggingFace example](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq.ipynb), it achieves `96%` accuracy on `
financial_phrasebank`.

### Requirement


```shell
# Install RedCoast
pip install redco==0.4.23
```

### Usage

[//]: # (*Commands below are tested on 8 x 80Gb H100 machines. You may want to adjust some numbers based on your hardware.*)

#### [For Multi-Host] Prepare initialization
In multi-host environments, HuggingFace's `FlaxModel.init_weights()` function cannot utilize the CPU backend to load the model into CPU memory before sharding it to GPU/TPUs. Therefore, it is necessary to prepare a JAX format checkpoint in advance for multi-host execution,
e.g.,
```
python save_init_ckpt.py --model_name_or_path facebook/bart-base
```
The prepared ckpt would be saved into `./bart-base`.


#### Data Parallel Only
When `--n_model_shards=1`, it doesn't shard the model and runs data parallelism only, which usually applied on smaller models, e.g., t5-base
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 python main.py 
```
Regarding `XLA_PYTHON_CLIENT_MEM_FRACTION`, see [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for more details about Jax's GPU memory preallocation.

#### Multi-GPU Model Parallel
This code supports tensor parallel to accommodate large models, e.g.,
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 python main.py --n_model_shards 4 
```

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 

