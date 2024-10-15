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

import fire
import jax
from transformers import FlaxAutoModelForCausalLM
from redco import Deployer


def main(model_name_or_path='huggyllama/llama-13b'):
    deployer = Deployer(workdir=None, jax_seed=0)

    ckpt_dir = './' + model_name_or_path.split('/')[-1]
    deployer.log_info(ckpt_dir, title='Init CKPT Dir to Save')

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_name_or_path, from_pt=True)
        deployer.save_ckpt(ckpt_dir=ckpt_dir, params=model.params)


if __name__ == '__main__':
    fire.Fire(main)
