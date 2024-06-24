## Diffusion Models

This example implements finetuning diffusion models. 
It supports 
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) or customizing by yourself (`lambdalabs/naruto-blip-captions` by default)
* assigning a text-to-image diffusion model from [diffusers](https://github.com/huggingface/diffusers) (`stabilityai/stable-diffusion-2-1-base` by default)

### Requirement


```shell
# Install RedCoast
pip install redco==0.4.18
# Install torchvision/torch (cpu version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Usage

*Commands below are tested on 8 x 80Gb H100 machines. You may want to adjust some numbers based on your hardware.*

#### [For Multi-Host] Prepare initialization
In multi-host environments, HuggingFace's `FlaxModel.init_weights()` function cannot utilize the CPU backend to load the model into CPU memory before sharding it to GPU/TPUs. Therefore, it is necessary to prepare a JAX format checkpoint in advance for multi-host execution,
e.g.,
```
python save_init_ckpt.py --model_name_or_path stabilityai/stable-diffusion-2-1-base
```
The prepared ckpt would be saved into `./stable-diffusion-2-1-base`.


#### Data Parallel Only
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 python main.py 
```
Regarding `XLA_PYTHON_CLIENT_MEM_FRACTION`, see [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for more details about Jax's GPU memory preallocation.

#### Multi-GPU Model Parallel
This code supports tensor parallel to accommodate large diffusion models:
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 python main.py \
  --per_device_batch_size 10 \
  --global_batch_size 30 \
  --n_model_shards 8
```

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 

#### Multi-Node Running (SLURM)

Here is an example sbatch script under SLURM, running this code on `N` nodes.
```
#!/bin/bash
#SBATCH --job-name=stable-diffusion
#SBATCH --partition=xxx
#SBATCH --nodes=N
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

module load cuda/12.3 cuDNN/9.1 nvhpc/24.3

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun python main.py --host0_address ${master_addr} --n_model_shards 8 --global_batch_size 40 --n_local_devices 8 --per_device_batch_size 10
```
The sharding can go cross multiple hosts, e.g., `--n_model_shards 16` if one has 2 (or more) nodes.

### Results


| Training Data                                                     | Model Generation                                                                   | 
|-------------------------------------------------------------------|------------------------------------------------------------------------------------|
| a girl in a red dress with long brown hair                        | a man in the woods with a sword                                                    |
| <img src="results/a_girl_in_a_red_dress_with_long_brown_hair.jpg" alt="drawing" width="500"/> | <img src="results/a_man_in_the_woods_with_a_sword.jpg" alt="drawing" width="500"/> |
| a guy with a bandage on his face                        | a girl with long brown hair and blue eyes                                          |
| <img src="results/a_guy_with_a_bandage_on_his_face.jpg" alt="drawing" width="500"/> | <img src="results/a_girl_with_long_hair_and_a_green_jacket.jpg" alt="drawing" width="500"/> |

