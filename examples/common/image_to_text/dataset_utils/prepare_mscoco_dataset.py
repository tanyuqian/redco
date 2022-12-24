import os
import fire
import json
import datasets


COCO17_LINKS = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip',
    'http://images.cocodataset.org/zips/test2017.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    'http://images.cocodataset.org/annotations/image_info_test2017.zip']


def download_raw_data(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for link in COCO17_LINKS:
        os.system(f'wget -P {save_dir} {link}')


def main(save_dir='mscoco_data'):
    download_raw_data(save_dir=f'{save_dir}/raw')

    ds = datasets.load_dataset(
        "ydshieh/coco_dataset_script",
        "2017",
        data_dir=os.path.abspath(f'{save_dir}/raw'),
        cache_dir=f'{save_dir}/cache')

    os.makedirs(f'{save_dir}/processed')

    with open(f'{save_dir}/processed/train.jsonl', 'w') as f:
        for example in ds['train']:
            example = {
                'image_path': example['image_path'],
                'caption': example['caption']
            }
            f.write(json.dumps(example) + '\n')

    with open(f'{save_dir}/processed/dev.jsonl', 'w') as f:
        for i in range(len(ds['validation']) // 2):
            example = ds['validation'][i]
            example = {
                'image_path': example['image_path'],
                'caption': example['caption']
            }
            f.write(json.dumps(example) + '\n')

    with open(f'{save_dir}/processed/test.jsonl', 'w') as f:
        for i in range(len(ds['validation']) // 2, len(ds['validation'])):
            example = ds['validation'][i]
            example = {
                'image_path': example['image_path'],
                'caption': example['caption']
            }
            f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    fire.Fire(main)