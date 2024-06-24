import fire
import jax
from transformers import FlaxCLIPTextModel
from diffusers import FlaxAutoencoderKL, FlaxUNet2DConditionModel
from redco import Deployer


def main(model_name_or_path='stabilityai/stable-diffusion-2-1-base'):
    deployer = Deployer(workdir=None, jax_seed=0)

    ckpt_dir = './' + model_name_or_path.split('/')[-1]
    deployer.log_info(ckpt_dir, title='Init CKPT Dir to Save')

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        text_encoder = FlaxCLIPTextModel.from_pretrained(
            model_name_or_path, subfolder="text_encoder", from_pt=True)
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            model_name_or_path, subfolder="vae", from_pt=True)
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            model_name_or_path, subfolder="unet", from_pt=True)
        params = {
            'text_encoder': text_encoder.params,
            'unet': unet_params,
            'vae': vae_params
        }

        deployer.save_ckpt(ckpt_dir=ckpt_dir, params=params)


if __name__ == '__main__':
    fire.Fire(main)
