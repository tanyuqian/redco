## Redco: A Lightweight Tool to Automate Distributed Training

Redco is a lightweight and user-friendly tool designed to automate distributed training and inference for large models while simplifying the ML pipeline development process without necessitating MLSys expertise from users.

* Redco allows for the simple implementation of distributed training and inference, eliminating the need for additional coding efforts or complex configurations, but still exhibits efficiency comparable to the most advanced model parallel tools.
* Redco enables customization of arbitrary ML pipelines within three functions, eliminating repetitive ans boilerplate coding, such as multi-host related processing, etc. We demonstrate that this mechanism is widely applicable to various ML algorithms

![](https://bowentan.bitcron.com/redco_coding.png)

### Features

* **Lightweight concepts**: *Redco* only introduces three concepts: Deployer, Trainer, and Predictor. You can be an expert in a couple of minites!
* **Easy-to-use**: Customize your pipeline with a couple of functions, each with a handful of lines. Designing your pipeline is the only thing you need to take care with *redco*.
* **Automatic deployment**: No need to take care of your multi-host or multi-device environment. *Redco* processes your environment automatically, as well as other pipeline-unrelated things, e.g., randomness, logging, etc.
* **Automatic model/data parallelism**: No need to concern your large models and large datasets. *Redco* distributes your models and datasets to all your devices automatically. 
* **No need to know JAX**: *Redco* only needs a couple of numpy-like functions as your pipeline design. 

### Installation

#### Install Jax
```
pip install --upgrade jax[cuda11_pip]==0.4.16 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Jax version (`==0.4.16`) can be flexible, as long as it matches your CUDA/CUDNN version.

If you are using TPU/CPU/AMD/Apple, see [here](https://github.com/google/jax#installation) for corresponding installation commands.

#### Install Redco
```
pip install redco
```



### Examples

Examples across a set of paradigms can be found in [examples/](examples/), including

* [classification/regression (GLUE & MNIST)](examples%2Fclassification_regression)
* [faderated learning (FedAvg)](examples%2Ffaderated_learning)
* [image to text (image captioning)](examples%2Fimage_to_text)
* [language modeling (Training LLMs like LLaMA)](examples%2Flanguage_modeling)
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

## Acknowledgement


The name of this package, *Redco*, is inspired by *Red Coast Base*, a key location in the story of Three-Body. From Red Coast Base, humanity broadcasts its first message into the vast universe. We thank Cixin Liu for such a masterpiece!

![](https://preview.redd.it/vonp0gvw6sd61.jpg?width=1680&format=pjpg&auto=webp&s=ec76245e86fe1cdc70bad33adddb6794b5176051)
(image: https://nhz123.lofter.com/post/1d7b3012_1c711d54c)