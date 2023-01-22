import os
import glob

import numpy as np
from PIL import Image
from torchvision import transforms


def get_dreambooth_dataset(predictor,
                           per_device_batch_size,
                           instance_dir,
                           instance_desc,
                           class_dir,
                           class_desc,
                           n_instance_samples_per_epoch,
                           n_class_samples_per_epoch,
                           with_prior_preservation,
                           text_key,
                           image_key):
    dataset = {'train': []}

    if with_prior_preservation:
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            examples_to_predict = \
                [{text_key: f'a photo of {class_desc}'}] * \
                n_class_samples_per_epoch
            images = predictor.predict(
                examples=examples_to_predict,
                per_device_batch_size=per_device_batch_size)

            for i, image in enumerate(images):
                image.save(f'{class_dir}/gen_{i}.jpg')
        else:
            assert len(glob.glob(f'{class_dir}/*')) >= n_class_samples_per_epoch

        for class_image_path in \
                glob.glob(f'{class_dir}/*')[:n_class_samples_per_epoch]:
            dataset['train'].append({
                image_key: Image.open(class_image_path),
                text_key: f'a photo of {class_desc}'
            })

    instance_paths = glob.glob(f'{instance_dir}/*')
    for idx in range(n_instance_samples_per_epoch):
        dataset['train'].append({
            image_key: Image.open(instance_paths[idx % len(instance_paths)]),
            text_key: f'a photo of {instance_desc}'
        })

    dataset['validation'] = [
        {text_key: f'a {instance_desc} in the Acropolis'},
        {text_key: f'a {instance_desc} is swimming'},
        {text_key: f'a {instance_desc} in a doghouse'},
        {text_key: f'a {instance_desc} is sleeping'},
        {text_key: f'a {instance_desc} in a bucket'},
        {text_key: f'a {instance_desc} getting a haircut'},
        {text_key: f'a {instance_desc} seen from the top'},
        {text_key: f'a {instance_desc} seen from the bottom'},
        {text_key: f'a {instance_desc} seen from the side'},
        {text_key: f'a {instance_desc} seen from the back'},
        {text_key: f'a depressed {instance_desc}'},
        {text_key: f'a sleeping {instance_desc}'},
        {text_key: f'a sad {instance_desc}'},
        {text_key: f'a joyous {instance_desc}'},
        {text_key: f'a barking {instance_desc}'},
        {text_key: f'a crying {instance_desc}'},
        {text_key: f'a frowning {instance_desc}'},
        {text_key: f'a screaming {instance_desc}'},
    ]

    print(f'#train:', len(dataset['train']))
    print(f'#validation:', len(dataset['validation']))

    return dataset


def dreambooth_image_preprocess_fn(image, resolution):
    image_transforms = transforms.Compose([
        transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    return np.array(image_transforms(image), dtype=np.float16)
