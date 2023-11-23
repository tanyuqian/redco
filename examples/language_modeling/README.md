## Language Modeling

This example implements instruction finetuning of Causal LLMs.
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) (Alpaca by default)
* assigning a causal language model from [transformers](https://github.com/huggingface/transformers) (LLaMA by default)
* multi-host running

### Requirement
Install Redco
```shell
pip install redco==0.4.12
```

### Use

```
python main.py \
    --model_name_or_path princeton-nlp/Sheared-LLaMA-1.3B \
    --n_epochs 3 \
    --per_device_batch_size 8 \
    --eval_per_device_batch_size 16 \
    --accumulate_grad_batches=4 \
    --max_length 512 \
    --eval_src_length 256 \
    --n_model_shards 4 
```
* `--model_name_or_path`: name or path of a CausalLM on HuggingFace, e.g., `huggyllama/llama-7b`.
* `--max_length`: total length of instruction + response in training. 
* `--eval_src_length`: length of instruction in inference.
* `--n_model_shards`: number of pieces to split your large model, `1` by default (pure data parallelism).


#### For Multi-host Envs
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


### Use saved params in HuggingFace 

```python
import fire
from flax.serialization import msgpack_restore
from transformers import AutoConfig, AutoTokenizer
from modeling_flax_llama import FlaxLlamaForCausalLM


def main(model_name_or_path='princeton-nlp/Sheared-LLaMA-1.3B',
         msgpack_filepath='./workdir/ckpts/params_last.msgpack'):
    model = FlaxLlamaForCausalLM(
        config=AutoConfig.from_pretrained(model_name_or_path), _do_init=False)
    params = msgpack_restore(open(msgpack_filepath, 'rb').read())
    
    # if save as a HuggingFace dir
    # model.save_pretrained('finetuned_llama', params=params)
    
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