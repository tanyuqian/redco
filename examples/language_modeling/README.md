## Language Modeling

### Requirement
Install Redco
```shell
pip install redco==0.4.17
```

### Download K2 Data Chunk
```
wget https://huggingface.co/datasets/LLM360/K2Datasets/resolve/main/chunk_379.jsonl
```

### Training
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.92 python main.py --n_model_shards 8
```