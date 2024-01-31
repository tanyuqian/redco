## Sequence-to-Sequence

This example implements seq2seq with Redco. 
It supports 
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) or customizing by yourself
* assigning a seq2seq model from [transformers](https://github.com/huggingface/transformers) (bart-base by default)
* multi-host running

### Requirement

Install Redco
```shell
pip install redco==0.4.15
```

### Usage

```shell
python main.py \
  --dataset_name xsum \
  --model_name_or_path google/flan-t5-xl \
  --n_model_shards 8
```
* `--n_model_shards`: number of pieces to split your large model, 1 by default (pure data parallelism). 

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 


#### For Multi-host Envs

##### General Case
```
python main.py \
    --host0_address 192.168.0.1 \ 
    --n_processes 2 \
    --process_id 1 \
    --n_local_devices 4
```
* `--n_processes`: number of hosts.
* `--host0_address`: the ip of host 0.
* `--process_id`: id of the current host (should vary across all hosts).
* `--n_local_devices`: number of devices (e.g., GPUs) on the machine. (Only required on some special envs, e.g., SLURM) 

##### Under SLURM
If you are using Redco under SLURM, just leave `n_processes` to be `None`. 
Below is an example of `run.sh` to submit, e.g., `sbatch run.sh`.

```shell
#!/bin/bash
#SBATCH --job-name=red_coast
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

srun python main.py --host0_address ${master_addr} --n_local_devices 4
```
* `--host0_address`: the ip of node 0 among your assigned nodes.
* `--n_local_devices`: number of GPUs on every machine. 
