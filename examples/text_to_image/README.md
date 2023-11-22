## Diffusion Models

This example implements finetuning diffusion models. 
It supports 
* assigning a dataset from [datasets](https://github.com/huggingface/datasets) or customizing by yourself.
* assigning a text-to-image diffusion model from [diffusers](https://github.com/huggingface/diffusers) (stable-diffusion-v1.4 by default)

### Requirement

Install Redco
```shell
pip install redco==0.4.11
```

### Usage

```shell
python main.py \
    --dataset_name lambdalabs/pokemon-blip-captions \
    --model_name_or_path duongna/stable-diffusion-v1-4-flax \
    --n_epochs 3 \
    --n_infer_steps 70 \
    ...
```

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 

