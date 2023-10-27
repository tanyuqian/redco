## Image-to-Text

This example implements image-to-text with Redco. 

### Requirement

Install Redco
```shell
pip install redco==0.4.8
```

### Usage

#### Download MSCOCO data

```
bash download_mscoco_dataset.sh
```

#### Training

```shell
python main.py \
  --data_dir=./mscoco_data \
  --model_name_or_path=nlpconnect/vit-gpt2-image-captioning \
  --per_device_batch_size 8 \
  --num_beams 4
``` 

See `def main(...)` in [main.py](main.py) for all the tunable arguments. 

#### Customize image encoder and text decoder

See [this HuggingFace example scrips](https://github.com/huggingface/transformers/blob/main/examples/flax/image-captioning/create_model_from_encoder_decoder_models.py#L85) to customize your model.