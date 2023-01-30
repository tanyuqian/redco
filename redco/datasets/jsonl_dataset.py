import json
from glob import glob

from .dataset import Dataset


class JsonlDataset(Dataset):
    def __init__(self, data_dir):
        self._split_filenames = {}
        for filename in glob(f'{data_dir}/*.jsonl'):
            split = filename.split('/')[-1][:-len('.jsonl')]
            self._split_filenames[split] = filename

    def __getitem__(self, split):
        examples = []
        for line in open(self._split_filenames[split]):
            examples.append(json.loads(line))

        return examples
