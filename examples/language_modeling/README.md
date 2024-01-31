## Language Modeling

This example implements instruction finetuning of Causal LLMs.
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) (Alpaca by default)
* assigning a causal language model from [transformers](https://github.com/huggingface/transformers) (LLaMA by default)
* multi-host running

### Requirement
Install Redco
```shell
pip install redco==0.4.15
```

### Use

```
XLA_PYTHON_CLIENT_MEM_FRACTION=.92 python main.py \
    --model_name_or_path princeton-nlp/Sheared-LLaMA-1.3B \
    --n_epochs 3 \
    --per_device_batch_size 8 \
    --eval_per_device_batch_size 16 \
    --accumulate_grad_batches 1 \
    --computation_dtype float32 \
    --max_length 512 \
    --eval_src_length 256 \
    --n_model_shards 4 
```
* `XLA_PYTHON_CLIENT_MEM_FRACTION=.92` *(Optional)*: can adjust the proportion of pre-allocated GPU memory to JAX.
* `--model_name_or_path`: name or path of a CausalLM on HuggingFace, e.g., `huggyllama/llama-7b` / `mistralai/Mistral-7B-v0.1`.
* `--computation_dtype`: dtype for model computation (might be different from dtype of parameters), `float32` by default.
* `--max_length`: total length of instruction + response in training. 
* `--eval_src_length`: length of instruction in inference.
* `--n_model_shards`: number of pieces to split your large model, `1` by default (pure data parallelism).

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
* `--n_local_devices`: devices on the machine. (Only required on some special envs, e.g., SLURM) 

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



### Use saved params  

#### Option 1. Run distributed generation

```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.92 python generate.py \
    --model_name_or_path huggyllama/llama-7b \
    --params_dir ./workdir/ckpts/last \
    --per_device_batch_size 8 \
    --n_model_shards 1 \
    --computation_dtype float32
```
* `XLA_PYTHON_CLIENT_MEM_FRACTION=.92` *(Optional)*: can adjust the proportion of pre-allocated GPU memory to JAX.
* `--model_name_or_path`: name or path of a CausalLM on HuggingFace, e.g., `huggyllama/llama-7b` / `mistralai/Mistral-7B-v0.1`.
* `--params_dir`: the path to saved params. If it's `None`, the pretrained model weights will be used. 
* `--n_model_shards`: number of pieces to split your large model, `1` by default (pure data parallelism).
* `--computation_dtype`: dtype for model computation (might be different from dtype of parameters), `float32` by default. 

See `def main(...)` in [generate.py](generate.py) for all the tunable arguments. See line 116 of [generate.py](generate.py) for changing dtype of inference parameters. 

#### Option 2. Load into a HuggingFace model (PyTorch)

```python
import fire
from flax.serialization import msgpack_restore
from transformers import AutoConfig, FlaxAutoModelForCausalLM, AutoModelForCausalLM

def main(model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
         msgpack_filepath='./workdir/ckpts/params_last.msgpack'):
    flax_model = FlaxAutoModelForCausalLM(
        config=AutoConfig.from_pretrained(model_name_or_path), _do_init=False)
    params = msgpack_restore(open(msgpack_filepath, 'rb').read())
    
    flax_model.save_pretrained('./saved_model_hf', params=params)
    
    pytorch_model = AutoModelForCausalLM.from_pretrained('./saved_model_hf', from_flax=True)

if __name__ == '__main__':
    fire.Fire(main)
```

#### Option 3. Load into a HuggingFace model (Jax/Flax)

```python
import fire
from flax.serialization import msgpack_restore
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModelForCausalLM

def main(model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
         msgpack_filepath='./workdir/ckpts/params_last.msgpack'):
    model = FlaxAutoModelForCausalLM(
        config=AutoConfig.from_pretrained(model_name_or_path), _do_init=False)
    params = msgpack_restore(open(msgpack_filepath, 'rb').read())
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    inputs = tokenizer(
        'Below is an instruction that describes a task. '
        'Write a response that appropriately completes the request. '
        '### Instruction: Find the capital of Spain. ### Response:',
        return_tensors='np')

    preds = model.generate(
        **inputs, params=params, do_sample=True, top_p=0.95, max_new_tokens=256
    ).sequences
    print(tokenizer.batch_decode(preds, skip_special_tokens=True))
    # ['Below is an instruction that describes a task. Write a response that appropriately completes the request. '
    #  '### Instruction: Find the capital of Spain. '
    #  '### Response: Madrid.']
    

if __name__ == '__main__':
    fire.Fire(main)
```



