from datasets import load_dataset


def get_dreambooth_dataset(predictor,
                           per_device_batch_size,
                           instance_desc,
                           class_desc,
                           n_instance_samples_per_epoch,
                           n_class_samples_per_epoch,
                           with_prior_preservation,
                           text_key,
                           image_key):
    dataset = {'train': []}

    if with_prior_preservation:
        examples_to_predict = \
            [{text_key: f'a photo of {class_desc}'}] * n_class_samples_per_epoch
        images = predictor.predict(
            examples=examples_to_predict,
            per_device_batch_size=per_device_batch_size)

        for image in images:
            dataset['train'].append({
                image_key: image,
                text_key: f'a photo of {class_desc}'
            })

    instance_images = [example['image'] for example in load_dataset(
        'nielsgl/dreambooth-ace', split='train')]

    for idx in range(n_instance_samples_per_epoch):
        dataset['train'].append({
            image_key: instance_images[idx % len(instance_images)],
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


