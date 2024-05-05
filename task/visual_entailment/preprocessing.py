# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
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
            'premise': [],
            'hypothesis': [],
            'label': [],
            'domain_id': [],
        },
        'valid': {
            'image': [],
            'image_id': [],
            'premise': [],
            'hypothesis': [],
            'label': [],
            'domain_id': [],
        },
        'test': {
            'image': [],
            'image_id': [],
            'premise': [],
            'hypothesis': [],
            'label': [],
            'domain_id': [],
        },
    }

    for i in range(len(loaded_data['image'])):
        data_dict[loaded_data['split'][i]]['image'].append(loaded_data['image'][i])
        data_dict[loaded_data['split'][i]]['image_id'].append(loaded_data['image_id'][i])
        data_dict[loaded_data['split'][i]]['premise'].append(loaded_data['premise'][i])
        data_dict[loaded_data['split'][i]]['hypothesis'].append(loaded_data['hypothesis'][i])
        data_dict[loaded_data['split'][i]]['label'].append(loaded_data['label'][i])
        data_dict[loaded_data['split'][i]]['domain_id'].append(loaded_data['domain_id'][i])

    check_path(os.path.join(args.preprocess_path, args.task_dataset))
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(args.preprocess_path, args.task_dataset, f'{split}_data.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

def load_data(args: argparse.Namespace):
    data_dict = {
        'image': [],
        'image_id': [],
        'premise': [],
        'hypothesis': [],
        'label': [],
        'domain_id': [], # 0: normal, realistic image,
        'split': []
    }

    # Load dataset
    if args.task_dataset == 'snli_ve':
        # First: download flickr30k-images-tar.gz from http://shannon.cs.illinois.edu/DenotationGraph/data/index.html
        raw_dataset = load_dataset('HuggingFaceM4/SNLI-VE', data_dir=os.path.abspath("./task/visual_entailment/")) # relative path - consider to change if necessary

        for split in ['train', 'validation', 'test']:
            for data in tqdm(raw_dataset[split]):
                data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                data_dict['image_id'].append(data['filename'])
                data_dict['premise'].append(data['premise'])
                data_dict['hypothesis'].append(data['hypothesis'])
                data_dict['label'].append(data['label'])
                data_dict['domain_id'].append(0) # 0: normal, realistic image
                if split == 'validation':
                    data_dict['split'].append('valid')
                else:
                    data_dict['split'].append(split)
    elif args.task_dataset == 'snli_ve_sports_ori':
        # First: download flickr30k-images-tar.gz from http://shannon.cs.illinois.edu/DenotationGraph/data/index.html
        raw_dataset = load_dataset('HuggingFaceM4/SNLI-VE', data_dir=os.path.abspath("./task/visual_entailment/"))
        sports_word_list = ['soccer', 'football']

        # Train data 4059 -> 8:1:1
        for data in tqdm(raw_dataset['train']):
            if any(word in data['premise'] for word in sports_word_list):
                data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                data_dict['image_id'].append(data['filename'])
                data_dict['premise'].append(data['premise'])
                data_dict['hypothesis'].append(data['hypothesis'])
                data_dict['label'].append(data['label'])
                data_dict['domain_id'].append(0) # 0: normal, realistic image

        # Split the data into train/valid/test 8:1:1
        image_id_set = list(set(data_dict['image_id']))
        print(f'Number of unique image_id: {len(image_id_set)}')
        train_image_id_list = image_id_set[:int(len(image_id_set) * 0.8)]
        valid_image_id_list = image_id_set[int(len(image_id_set) * 0.8):int(len(image_id_set) * 0.9)]
        test_image_id_list = image_id_set[int(len(image_id_set) * 0.9):]

        # Assign split to each data
        for i in tqdm(range(len(data_dict['image_id']))):
            if data_dict['image_id'][i] in train_image_id_list:
                data_dict['split'].append('train')
            elif data_dict['image_id'][i] in valid_image_id_list:
                data_dict['split'].append('valid')
            elif data_dict['image_id'][i] in test_image_id_list:
                data_dict['split'].append('test')
            else:
                pass # Discard the data
    else:
        raise NotImplementedError(f'Not implemented dataset: {args.task_dataset}')

    assert len(data_dict['image']) == \
           len(data_dict['image_id']) == \
           len(data_dict['premise']) == \
           len(data_dict['hypothesis']) == \
           len(data_dict['label']) == \
           len(data_dict['domain_id']) == \
           len(data_dict['split'])

    return data_dict
