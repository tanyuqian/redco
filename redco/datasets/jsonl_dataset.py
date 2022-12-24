import json
from glob import glob

from .dataset import Dataset


class JsonlDataset(Dataset):
    def __init__(self, data_dir):
        self._split_filenames = {}
        for filename in glob(f'{data_dir}/*.jsonl'):
            split = filename.split('/')[-1][:-len('.jsonl')]
            self._split_filenames[split] = filename

    def get_examples(self, split):
        examples = []
        for line in open(self._split_filenames[split]):
            examples.append(json.loads(line))

        return examples

    def get_size(self, split):
        n_lines = 0
        for _ in open(self._split_filenames[split]):
            n_lines += 1
        return n_lines
