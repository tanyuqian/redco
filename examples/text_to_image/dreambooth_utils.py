import os
import glob
from PIL import Image


def get_dreambooth_dataset(predictor,
                           per_device_batch_size,
                           instance_dir,
                           instance_prompt,
                           class_dir,
                           class_prompt,
                           n_class_images,
                           text_key,
                           image_key):
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    if len(glob.glob(f'{class_dir}/*')) < n_class_images:
        examples_to_predict = [{text_key: class_prompt}] * (
                n_class_images - len(glob.glob(f'{class_dir}/*')))
        images = predictor.predict(
            examples=examples_to_predict,
            per_device_batch_size=per_device_batch_size)

        for i, image in enumerate(images):
            image.save(f'{class_dir}/gen_{i}.jpg')

    instance_paths = glob.glob(f'{instance_dir}/*')
    class_paths = glob.glob(f'{class_dir}/*')
    dataset = {'train': [], 'validation': []}
    for idx in range(max(len(instance_paths), n_class_images)):
        dataset['train'].append({
            image_key: Image.open(instance_paths[idx % len(instance_paths)]),
            text_key: instance_prompt
        })
        dataset['train'].append({
            image_key: Image.open(class_paths[idx % len(class_paths)]),
            text_key: class_prompt
        })

    for expression in ['depressed',
                       'sleeping',
                       'sad',
                       'joyous',
                       'barking',
                       'crying',
                       'frowning',
                       'screaming']:
        dataset['validation'].append(
            {text_key: f'A {expression} {instance_prompt}'})

    return dataset
