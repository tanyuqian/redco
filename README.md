![](images/redco_logo.png)

**Red Coast** (redco) is a lightweight and user-friendly tool designed to automate distributed training and inference for large models while simplifying the ML pipeline development process without necessitating MLSys expertise from users.

Check out our [Tech Report](https://arxiv.org/pdf/2310.16355.pdf) for details! 
Here is also a [Quick Tutorial](tutorials/quick.md) for you to become an expert of distributed training with Redco in several minutes!

* Redco allows for the simple implementation of distributed training and inference, eliminating the need for additional coding efforts or complex configurations, but still exhibits efficiency comparable to the most advanced model parallel tools.
* Redco enables customization of arbitrary ML pipelines within three functions, eliminating repetitive ans boilerplate coding, such as multi-host related processing, etc. We demonstrate that this mechanism is widely applicable to various ML algorithms
* The backend of Redco is based on JAX, but users doesn't need to be JAX experts. Knowing `numpy` is good enough!

![](images/redco_coding.png)

### Installation

#### Install Redco
```
pip install redco
```

#### Adjust Jax & Flax versions
The command above would automatically install cpu version of jax, so the version of Jax need to be adjusted based on your device. For example,
```
pip install --upgrade flax==0.7.0
pip install --upgrade jax[cuda11_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Jax version (`==0.4.13`) and Flax version (`==0.7.0`) can be flexible, as long as they match your CUDA/CUDNN/NCCL version. 
Besides, the Flax modeling in the HuggingFace implementation sometimes doesn't support the most recent Jax & Flax versions. 

If you are using TPU/CPU/AMD/Apple, see [here](https://github.com/google/jax#installation) for corresponding installation commands.


### Examples

Examples across a set of paradigms can be found in [examples/](examples/), including

* [Classification/regression (GLUE & MNIST)](examples%2Fclassification_regression)
* [Faderated learning (FedAvg)](examples%2Ffaderated_learning)
* [Image to text (Image captioning)](examples%2Fimage_to_text)
* [Language modeling (Instruction Tuning of LLMs)](examples%2Flanguage_modeling)
* [Meta learning (MAML)](examples%2Fmeta_learning)
* [Reinforcement learning (PPO & DDPG)](examples%2Freinforcement_learning)
* [Text to image (StableDiffusion)](examples%2Ftext_to_image)
* [Text to text (Seq2seq)](examples%2Ftext_to_text)

### Exemplar large model settings

The table below shows runnable model LLM finetuning on different kinds of servers. Numbers inside the brackets are the maximum length in training. All the settings are with full precision (fp32) and Adam optimizer.

| 2 $\times$ 1080Ti <br/>(2 $\times$ 10G) | 4 $\times$ A100 <br/>(4 $\times$ 40G) | 2 $\times$ TPU-v4 <br/>(2 hosts $\times$ 4 chips $\times$ 32G) | 16 $\times$ TPU-v4 <br/>(16 hosts $\times$ 4 chips $\times$ 32G) |
|-----------------------------------------|---------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------|
| BART-Large (1024)                       | LLaMA-7B (1024)                       | T5-XL-11B (512)                                                | OPT-66B (512)                                                    |
| GPT2-Large (512)                        | GPT-J-6B (1024)                       | OPT-13B (1024)                                                 |                                                                  |

Go to [example/language_modeling](examples%2Flanguage_modeling) and [examples/text_to_text](examples%2Ftext_to_text) to try them out!


## Reference

We now have a [paper](https://arxiv.org/pdf/2310.16355.pdf) you can cite for the Red Coast library:

```
Redco: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs
Bowen Tan, Yun Zhu, Lijuan Liu, Hongyi Wang, Yonghao Zhuang, Jindong Chen, Eric Xing, Zhiting Hu
Mlsys Workshop @ NeurIPS 2023

@article{tan2023redco,
  title={Redco: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs},
  author={Tan, Bowen and Zhu, Yun and Liu, Lijuan and Wang, Hongyi and Zhuang, Yonghao and Chen, Jindong and Xing, Eric and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16355},
  year={2023}
}
```

## Acknowledgement


The name of this package, *Redco*, is inspired by *Red Coast Base*, a key location in the story of Three-Body. From Red Coast Base, humanity broadcasts its first message into the vast universe. We thank Cixin Liu for such a masterpiece!

![](images/red_coast.png)