## Language Modeling

This example implements language modeling with Redco. 
Specifically, it runs instruction finetuning on LLaMA (or other LMs available in HuggingFace, e.g., GPT-J) with [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca). 

### Requirement
Install a Jax implementation of LLaMA [here](https://github.com/Sea-Snell/JAX_llama).
```shell
git clone https://github.com/Sea-Snell/JAX_llama.git
cd JAX_llama
pip install -e .
```

### Running

This example finetunes LLaMA-7B on a server with 4 A100s (40G each), which runs by
```shell
python main.py
```
All the default configs can be found inside ```def main(...)``` of [main.py](main.py).

If it reports OOM, we suggest firstly try updating the env variable ```XLA_PYTHON_CLIENT_MEM_FRACTION``` to pre-allocate more memory to Jax. 
See [Jax's notes](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for more details.

If you are running this code on other types of GPUs or other LMs, you may mainly tune batch size, model parallel, computation dtype, etc. For example  
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.92 python main.py \
  --model_name_or_path EleutherAI/gpt-j-6b \
  --bf16 True \  # use bfloat16 in computation
  --n_model_shards 4  \  # how many pieces (GPUs) to split your model into
  --per_device_batch_size 12 \ 
  --accumulate_grad_batches 8 
```
