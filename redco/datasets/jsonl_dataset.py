#  Copyright 2021 Google LLC
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from glob import glob
import tqdm

from .dataset import Dataset


class JsonlDataset(Dataset):
    def __init__(self, data_dir):
        self._split_filenames = {}
        for filename in glob(f'{data_dir}/*.jsonl'):
            split = filename.split('/')[-1][:-len('.jsonl')]
            self._split_filenames[split] = filename

    def __getitem__(self, split):
        examples = []
        for line in tqdm.tqdm(open(self._split_filenames[split]),
                              desc=f'loading {split} examples'):
            examples.append(json.loads(line))

        return examples
