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
import jax.numpy as jnp
from transformers import FlaxCLIPTextModel
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from redco import Deployer


def main(model_name_or_path='stabilityai/stable-diffusion-2-1-base'):
    deployer = Deployer(workdir=None, jax_seed=0)

    ckpt_dir = './' + model_name_or_path.split('/')[-1]
    deployer.log_info(ckpt_dir, title='Init CKPT Dir to Save')

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        text_encoder = FlaxCLIPTextModel.from_pretrained(
            model_name_or_path,
            subfolder="text_encoder", from_pt=True, dtype=jnp.float16)
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            model_name_or_path,
            subfolder="vae", from_pt=True, dtype=jnp.float16)
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            model_name_or_path,
            subfolder="unet", from_pt=True, dtype=jnp.float32)
        params = {
            'text_encoder': text_encoder.params,
            'unet': unet_params,
            'vae': vae_params
        }

        deployer.save_ckpt(ckpt_dir=ckpt_dir, params=params)


if __name__ == '__main__':
    fire.Fire(main)
