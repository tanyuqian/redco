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

    dataset = []
    for image_path in glob.glob(f'{instance_dir}/*'):
        dataset.append({
            text_key: instance_prompt,
            image_key: Image.open(image_path)
        })

    for image_path in glob.glob(f'{class_dir}/*'):
        dataset.append({
            text_key: class_prompt,
            image_key: Image.open(image_path)
        })

    return dataset
