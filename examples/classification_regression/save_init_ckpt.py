import fire
import jax
from transformers import FlaxAutoModelForSequenceClassification
from redco import Deployer


def main(model_name_or_path='FacebookAI/roberta-large', num_labels=2):
    deployer = Deployer(workdir=None, jax_seed=0)

    ckpt_dir = './' + model_name_or_path.split('/')[-1]
    deployer.log_info(ckpt_dir, title='Init CKPT Dir to Save')

    with jax.default_device(jax.local_devices(backend='cpu')[0]):
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, from_pt=True)
        deployer.save_ckpt(ckpt_dir=ckpt_dir, params=model.params)


if __name__ == '__main__':
    fire.Fire(main)
