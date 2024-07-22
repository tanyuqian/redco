![](images/redco_logo.png)

**Red Coast** (redco) is a lightweight and user-friendly tool designed to automate distributed training and inference for large models while simplifying the ML pipeline development process without necessitating MLSys expertise from users.

Check out our [Tech Report](https://aclanthology.org/2024.naacl-demo.14/) for more details! 

**RedCoast: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs** \
Bowen Tan, Yun Zhu, Lijuan Liu, Hongyi Wang, Yonghao Zhuang, Jindong Chen, Eric Xing, Zhiting Hu \
NAACL 2024, Demo / MLSys Workshop @ NeurIPS 2023 \
[[Paper]](https://aclanthology.org/2024.naacl-demo.14/) 
[[Twitter]](https://x.com/BowenTan8/status/1730240627068031295?s=20) 
[[Slides]](https://drive.google.com/file/d/1MmBjxP5gInqhg0ydasby2a5UauLZFxQH/view) 
[[Demo Video]](https://bowentan.bitcron.com/RedCoast_demo.webm) \
<span style="color:red">(Best Demo Paper Runner Up @ NAACL 2024)</span>

RedCoast supports *Large Models* + *Complex Algorithms*, in a *lightweight* and *user-friendly* manner: 
* Large Models beyond Transformers, e.g, [Stable Diffusion](examples/text_to_image), etc.
* Complex algorithms beyond cross entropy, e.g., [Meta Learning](examples/meta_learning), etc.

![](images/redco_coding.png)

### Installation

#### Install RedCoast
```
pip install redco
```

#### Adjust Jax to GPU/TPU version
The command above would automatically install cpu version of jax, so the version of Jax need to be adjusted based on your device. 
For example, on GPUs,
```
# for cuda-12.x
pip install --upgrade "jax[cuda12]"
# for cuda-11.x
pip install --upgrade jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
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
```
@inproceedings{tan-etal-2024-redcoast,
    title = "{R}ed{C}oast: A Lightweight Tool to Automate Distributed Training of {LLM}s on Any {GPU}/{TPU}s",
    author = "Tan, Bowen  and
      Zhu, Yun  and
      Liu, Lijuan  and
      Wang, Hongyi  and
      Zhuang, Yonghao  and
      Chen, Jindong  and
      Xing, Eric  and
      Hu, Zhiting",
    editor = "Chang, Kai-Wei  and
      Lee, Annie  and
      Rajani, Nazneen",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-demo.14",
    pages = "137--147",
    abstract = "The recent progress of AI can be largely attributed to large language models (LLMs). However, their escalating memory requirements introduce challenges for machine learning (ML) researchers and engineers. Addressing this requires developers to partition a large model to distribute it across multiple GPUs or TPUs. This necessitates considerable coding and intricate configuration efforts with existing model parallel tools, such as Megatron-LM, DeepSpeed, and Alpa. These tools require users{'} expertise in machine learning systems (MLSys), creating a bottleneck in LLM development, particularly for developers without MLSys background. In this work, we present RedCoast (Redco), a lightweight and user-friendly tool crafted to automate distributed training and inference for LLMs, as well as to simplify ML pipeline development. The design of Redco emphasizes two key aspects. Firstly, to automate model parallelism, our study identifies two straightforward rules to generate tensor parallel strategies for any given LLM. Integrating these rules into Redco facilitates effortless distributed LLM training and inference, eliminating the need of additional coding or complex configurations. We demonstrate the effectiveness by applying Redco on a set of LLM architectures, such as GPT-J, LLaMA, T5, and OPT, up to the size of 66B. Secondly, we propose a mechanism that allows for the customization of diverse ML pipelines through the definition of merely three functions, avoiding redundant and formulaic code like multi-host related processing. This mechanism proves adaptable across a spectrum of ML algorithms, from foundational language modeling to complex algorithms like meta-learning and reinforcement learning. As a result, Redco implementations exhibit significantly fewer lines of code compared to their official counterparts. RedCoast (Redco) has been released under Apache 2.0 license at https://github.com/tanyuqian/redco.",
}
```

## Acknowledgement


The name of this package is inspired by *Red Coast Base*, a key location in the story of Three-Body. From Red Coast Base, humanity broadcasts its first message into the vast universe. We thank Cixin Liu for such a masterpiece!

![](images/red_coast.png)