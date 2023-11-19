## Classification/Regression

This example implements classification/regression with Redco. 
It supports 
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) for customize
* assigning a classfication model from [transformers](https://github.com/huggingface/transformers) 
* multi-host running

### Requirement

Install Redco
```shell
pip install redco==0.4.11
```

### Usage

```shell
python main.py \
    --dataset_name sst2 \
    --model_name_or_path roberta-large \
    --n_model_shards 2
```
* `--n_model_shards`: number of pieces to split your large model, 1 by default (pure data parallelism). 

See `def main(...)` in [glue_main.py](glue_main.py) for all the tunable arguments. 


#### For Multi-host Envs
```
python glue_main.py \
    --n_processes 2 \
    --host0_address 192.168.0.1 \ 
    --process_id 1 \
...
```
* `--n_processes`: number of hosts.
* `--host0_address`: the ip of host 0 with an arbitrary available port number.
* `--process_id`: id of the current host (should vary across all hosts).
