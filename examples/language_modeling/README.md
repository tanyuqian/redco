## Language Modeling

This example implements training causal language models, supporting causal language models from [transformers](https://github.com/huggingface/transformers) (`huggyllama/llama-13b` by default)

### Requirement


```shell
# Install RedCoast
pip install redco==0.4.23
# Install torchvision/torch (cpu version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Usage

*Commands below are tested on 8 x 80Gb H100 machines. You may want to adjust some numbers based on your hardware.*

#### [For Multi-Host] Prepare initialization
In multi-host environments, HuggingFace's FlaxModel.init_weights() function cannot utilize the CPU backend to load the model into CPU memory before sharding it to GPU/TPUs. Therefore, it is necessary to prepare a JAX format checkpoint in advance for multi-host execution,
e.g.,
```
python make_init_ckpt.py --model_name_or_path huggyllama/llama-13b
```
The prepared ckpt would be saved into `./llama-13b`.


#### Data Parallel Only
When `--n_model_shards=1`, it doesn't shard the model and runs data parallelism only, which usually applied on smaller models, e.g., GPT-2.
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python main.py --model_name_or_path gpt2 --n_model_shards 1
```
Regarding `XLA_PYTHON_CLIENT_MEM_FRACTION`, see [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for more details about Jax's GPU memory preallocation.

#### Multi-GPU Model Parallel
This code supports tensor parallel to accommodate large diffusion models:
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python main.py 
```

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 

#### Multi-Node Running (SLURM)

Here is an example sbatch script under SLURM training [LLM360/K2-Chat (65B)](https://huggingface.co/LLM360/K2-Chat) on 8 nodes, each with 8 H100s.
* The sharding can go cross multiple hosts, e.g., `--n_model_shards 32` here means the model is splitted into 32 shards (every data parallel takes 4 nodes).
* For longer context length, the model should be splitted into more shards, e.g., `--context_length 8192` needs `--n_model_shards 64`.

```
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --exclusive

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export XLA_PYTHON_CLIENT_MEM_FRACTION=.96
export XLA_FLAGS="--xla_gpu_shard_autotuning=false"

ulimit -n 4096

# python make_init_ckpt.py --model_name_or_path LLM360/K2-Chat

srun python main.py \
    --host0_address ${master_addr} \
    --n_local_devices 8 \
    --model_name_or_path LLM360/K2-Chat \
    --init_ckpt_dir ./K2-Chat \
    --n_model_shards 32 \
    --global_batch_size 32 \
    --per_device_batch_size 2 \
    --context_length 2048 \
    --n_epochs 10
```
