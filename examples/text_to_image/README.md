## Diffusion Models

This example implements finetuning diffusion models. 
It supports 
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) or customizing by yourself (`lambdalabs/naruto-blip-captions` by default)
* assigning a text-to-image diffusion model from [diffusers](https://github.com/huggingface/diffusers) (`stabilityai/stable-diffusion-2-1-base` by default)

### Requirement


```shell
# Install RedCoast
pip install redco==0.4.17
# Install torchvision/torch (cpu version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Usage

*Commands below are tested on 8 x 80Gb H100 machines. You may want to adjust some numbers based on your hardware.*

#### Data Parallel Only
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 python main.py 
```

#### Multi-GPU Model Parallel
This code supports tensor parallel:
```shell
XLA_PYTHON_CLIENT_MEM_FRACTION=.80 python main.py \
  --per_device_batch_size 10 \
  --global_batch_size 30 \
  --n_model_shards 8
```

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 

