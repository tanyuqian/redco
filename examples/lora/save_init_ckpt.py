import fire
import jax
from transformers import FlaxAutoModelForSeq2SeqLM
from redco import Deployer


def main(model_name_or_path='google/flan-t5-xxl'):
    deployer = Deployer(workdir=None, jax_seed=0)

    ckpt_dir = './' + model_name_or_path.split('/')[-1]
    deployer.log_info(ckpt_dir, title='Init CKPT Dir to Save')

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, from_pt=True)
        deployer.save_ckpt(ckpt_dir=ckpt_dir, params=model.params)


if __name__ == '__main__':
    fire.Fire(main)
