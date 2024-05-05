# Standard Library Modules
import os
import sys
import json
import pickle
import requests
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from PIL import Image
# Huggingface Modules
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def preprocessing(args: argparse.Namespace):
    loaded_data = load_data(args)

    data_dict = {
        'train': {
            'image': [],
            'image_id': [],
            'caption': [],
            'caption_number': [],
            'all_captions': [],
            'domain_id': [],
        },
        'valid': {
            'image': [],
            'image_id': [],
            'caption': [],
            'caption_number': [],
            'all_captions': [],
            'domain_id': [],
        },
        'test': {
            'image': [],
            'image_id': [],
            'caption': [],
            'caption_number': [],
            'all_captions': [],
            'domain_id': [],
        }
    }

    for i in tqdm(range(len(loaded_data['image']))):
        data_dict[loaded_data['split'][i]]['image'].append(loaded_data['image'][i])
        data_dict[loaded_data['split'][i]]['image_id'].append(loaded_data['image_id'][i])
        data_dict[loaded_data['split'][i]]['caption'].append(loaded_data['caption'][i])
        data_dict[loaded_data['split'][i]]['caption_number'].append(loaded_data['caption_number'][i])
        data_dict[loaded_data['split'][i]]['all_captions'].append(loaded_data['all_captions'][i])
        data_dict[loaded_data['split'][i]]['domain_id'].append(loaded_data['domain_id'][i])

    check_path(os.path.join(args.preprocess_path, args.task_dataset))
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(args.preprocess_path, args.task_dataset, f'{split}_data.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

def load_data(args: argparse.Namespace):
    data_dict = {
        'image': [],
        'image_id': [],
        'caption': [],
        'caption_number': [],
        'all_captions': [],
        'domain_id': [], # 0: normal, realistic image,
        'split': []
    }

    # Load dataset
    if args.task_dataset == 'flickr8k':
        raw_dataset = load_dataset('jxie/flickr8k')

        for split in ['train', 'validation']:
            for data in tqdm(raw_dataset[split]):
                all_caption = [data['caption_0'], data['caption_1'], data['caption_2'], data['caption_3'], data['caption_4']]
                for i in range(5):
                    data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                    data_dict['image_id'].append(None)
                    data_dict['caption'].append(data[f'caption_{i}'])
                    data_dict['caption_number'].append(i)
                    data_dict['all_captions'].append(all_caption)
                    data_dict['domain_id'].append(0) # 0: normal, realistic image
                    if split == 'validation':
                        data_dict['split'].append('valid')
                    else:
                        data_dict['split'].append(split)
        for data in tqdm(raw_dataset['test']):
            all_caption = [data['caption_0'], data['caption_1'], data['caption_2'], data['caption_3'], data['caption_4']]
            data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
            data_dict['image_id'].append(None)
            data_dict['caption'].append(data['caption_0'])
            data_dict['caption_number'].append(0)
            data_dict['all_captions'].append(all_caption)
            data_dict['domain_id'].append(0) # 0: normal, realistic image
            data_dict['split'].append('test')
    elif args.task_dataset == 'flickr30k':
        raw_dataset = load_dataset('nlphuji/flickr30k')

        for data in tqdm(raw_dataset['test']):
            if data['split'] in ['train', 'val']:
                for i in range(len(data['sentids'])):
                    data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                    data_dict['image_id'].append(data['filename'])
                    data_dict['caption'].append(data['caption'][i])
                    data_dict['caption_number'].append(i)
                    data_dict['all_captions'].append(data['caption'])
                    data_dict['domain_id'].append(0) # 0: normal, realistic image
                    if data['split'] == 'val':
                        data_dict['split'].append('valid')
                    else:
                        data_dict['split'].append(data['split'])
            elif data['split'] == 'test':
                data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                data_dict['image_id'].append(data['filename'])
                data_dict['caption'].append(data['caption'][0])
                data_dict['caption_number'].append(0)
                data_dict['all_captions'].append(data['caption'])
                data_dict['domain_id'].append(0)
                data_dict['split'].append('test')
    elif args.task_dataset == 'coco_karpathy':
        raw_dataset = load_dataset('yerevann/coco-karpathy')

        for split in ['train', 'validation']:
            for data in tqdm(raw_dataset[split]):
                image_url = data['url']
                # image = Image.open(requests.get(image_url, stream=True).raw) # uncomment this if you want to download images from url
                image_path = os.path.join(args.data_path, 'coco_2014', data['filepath'], data['filename']) # comment this if you didn't downloaded images manually
                image = Image.open(image_path).convert('RGB').resize((args.image_resize_size, args.image_resize_size))
                for i in range(len(data['sentids'])):
                    data_dict['image'].append(image)
                    data_dict['image_id'].append(data['filename'])
                    data_dict['caption'].append(data['sentences'][i])
                    data_dict['caption_number'].append(i)
                    data_dict['all_captions'].append(data['sentences'])
                    data_dict['domain_id'].append(0) # 0: normal, realistic image
                    if split == 'validation':
                        data_dict['split'].append('valid')
                    else:
                        data_dict['split'].append(split)
        for data in tqdm(raw_dataset['test']):
            image_url = data['url']
            # image = Image.open(requests.get(image_url, stream=True).raw) # uncomment this if you want to download images from url
            image_path = os.path.join(args.data_path, 'coco_2014', data['filepath'], data['filename'])
            image = Image.open(image_path).convert('RGB').resize((args.image_resize_size, args.image_resize_size))
            data_dict['image'].append(image)
            data_dict['image_id'].append(data['filename'])
            data_dict['caption'].append(data['sentences'][0])
            data_dict['caption_number'].append(0)
            data_dict['all_captions'].append(data['sentences'])
            data_dict['domain_id'].append(0)
            data_dict['split'].append('test')
    elif args.task_dataset == 'uit_viic_en_ori':
        with open('./task/captioning/UIT_VIIC_SPLIT.json', 'rb') as f:
            data = json.load(f)
            data = data['data']

        for each_data in tqdm(data):
            filepath = "train2014" if "train2014" in each_data['file_name'] else "val2014"
            image_path = os.path.join(args.data_path, 'coco_2014', filepath, each_data['file_name'])
            image = Image.open(image_path).convert('RGB').resize((args.image_resize_size, args.image_resize_size))

            if each_data['split'] in ['train', 'valid']:
                for i in range(len(each_data['captions'])):
                    data_dict['image'].append(image)
                    data_dict['image_id'].append(each_data['file_name'])
                    data_dict['caption'].append(each_data['captions'][i])
                    data_dict['caption_number'].append(i)
                    data_dict['all_captions'].append(each_data['captions'])
                    data_dict['split'].append(each_data['split'])
                    data_dict['domain_id'].append(0) # 0: normal, realistic image
            elif each_data['split'] == 'test':
                data_dict['image'].append(image)
                data_dict['image_id'].append(each_data['file_name'])
                data_dict['caption'].append(each_data['captions'][0])
                data_dict['caption_number'].append(0)
                data_dict['all_captions'].append(each_data['captions'])
                data_dict['split'].append(each_data['split'])
                data_dict['domain_id'].append(0)
    else:
        raise NotImplementedError(f'Not implemented dataset: {args.task_dataset}')

    assert len(data_dict['image']) == \
           len(data_dict['image_id']) == \
           len(data_dict['caption']) == \
           len(data_dict['caption_number']) == \
           len(data_dict['all_captions']) == \
           len(data_dict['domain_id']) == \
           len(data_dict['split'])

    return data_dict
