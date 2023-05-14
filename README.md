## Redco: Distributed LLM training with a single line of code

Redco is a user-friendly toolkit for developing and scaling up Jax/Flax-based pipelines, 
where you can define your pipeline in a couple of functions, without concerning environmental issues, 
e.g., multi-host, parallelization for large dataset and large models, etc. 

![](https://bowentan.bitcron.com/redco_framework.jpg)

### Features

* **Lightweight concepts**: *Redco* only introduces three concepts: Deployer, Trainer, and Predictor. You can be an expert in a couple of minites!
* **Easy-to-use**: Customize your pipeline with 4-5 functions, each with a handful of lines. Designing your pipeline is the only thing you need to take care with *redco*.
* **Automatic deployment**: No need to take care of your multi-host or multi-device environment. *Redco* processes your environment automatically, as well as other pipeline-unrelated things, e.g., randomness, logging, etc.
* **Automatic model/data parallelism**: No need to concern your large models and large datasets. *Redco* distributes your models and datasets to all your devices automatically. 
* **Easy to migrate from PyTorch**: No need to know complex Jax functions (e.g., ```pmap()```, ```pjit()```, etc.). *Redco* only needs a couple of numpy-like functions from you as your pipeline design. 

### Installation

Redco can be installed by
```
pip install redco
```
Make sure correct [Jax version](https://github.com/google/jax#installation) is installed according to your device. 



### Examples

Examples across a set of paradigms can be found in [examples/](examples/), including

* [classification/regression (GLUE & MNIST)](examples%2Fclassification_regression)
* [faderated learning (FedAvg)](examples%2Ffaderated_learning)
* [image to text (image captioning)](examples%2Fimage_to_text)
* [language modeling](examples%2Flanguage_modeling)
* [meta learning (MAML)](examples%2Fmeta_learning)
* [reinforcement learning (PPO & DDPG & MADDPG)](examples%2Freinforcement_learning)
* [text to image (StableDiffusion)](examples%2Ftext_to_image)
* [text to text (Seq2seq)](examples%2Ftext_to_text)

### Exemplar large model settings

The table below shows runnable model LLM finetuning on different kinds of servers. Numbers inside the brackets are the maximum length in training. All the settings are with full precision (fp32) and Adam optimizer.

| 2 $\times$ 1080Ti <br/>(2 $\times$ 10G) | 4 $\times$ A100 <br/>(4 $\times$ 40G) | 2 $\times$ TPU-v4 <br/>(2 hosts $\times$ 4 chips $\times$ 32G) | 16 $\times$ TPU-v4 <br/>(16 hosts $\times$ 4 chips $\times$ 32G) |
|-----------------------------------------|---------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------|
| BART-Large (1024)                       | LLaMA-7B (1024)                       | T5-XL-11B (512)                                                | OPT-66B (512)                                                    |
| GPT2-Large (512)                        | GPT-J-6B (1024)                       | OPT-13B (1024)                                                 |                                                                  |

Go to [example/language_modeling](examples%2Flanguage_modeling) and [examples/text_to_text](examples%2Ftext_to_text) to try them out!
