# Standard Library Modules
import os
import sys
import json
import pickle
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
            'question_id': [],
            'question': [],
            'answer': [],
            'full_answer': [],
            'question_type': [],
            'answer_type': [],
            'caption': [],
            'domain_id': [],
        },
        'valid': {
            'image': [],
            'image_id': [],
            'question_id': [],
            'question': [],
            'answer': [],
            'full_answer': [],
            'question_type': [],
            'answer_type': [],
            'caption': [],
            'domain_id': [],
        },
        'test': { # Only for UIT-VIIC-EN-ORI
            'image': [],
            'image_id': [],
            'question_id': [],
            'question': [],
            'answer': [],
            'full_answer': [],
            'question_type': [],
            'answer_type': [],
            'caption': [],
            'domain_id': [],
        },
        'testdev': {
            'image': [],
            'image_id': [],
            'question_id': [],
            'question': [],
            'answer': [],
            'full_answer': [],
            'question_type': [],
            'answer_type': [],
            'caption': [],
            'domain_id': [],
        },
        'teststd': {
            'image': [],
            'image_id': [],
            'question_id': [],
            'question': [],
            'answer': [],
            'full_answer': [],
            'question_type': [],
            'answer_type': [],
            'caption': [],
            'domain_id': [],
        },
    }

    for i in range(len(loaded_data['image'])):
        data_dict[loaded_data['split'][i]]['image'].append(loaded_data['image'][i])
        data_dict[loaded_data['split'][i]]['image_id'].append(loaded_data['image_id'][i])
        data_dict[loaded_data['split'][i]]['question_id'].append(loaded_data['question_id'][i])
        data_dict[loaded_data['split'][i]]['question'].append(loaded_data['question'][i])
        data_dict[loaded_data['split'][i]]['answer'].append(loaded_data['answer'][i])
        data_dict[loaded_data['split'][i]]['full_answer'].append(loaded_data['full_answer'][i])
        data_dict[loaded_data['split'][i]]['question_type'].append(loaded_data['question_type'][i])
        data_dict[loaded_data['split'][i]]['answer_type'].append(loaded_data['answer_type'][i])
        data_dict[loaded_data['split'][i]]['caption'].append(loaded_data['caption'][i])
        data_dict[loaded_data['split'][i]]['domain_id'].append(loaded_data['domain_id'][i])

    check_path(os.path.join(args.preprocess_path, args.task_dataset))
    for split in ['train', 'valid', 'test', 'testdev', 'teststd']:
        with open(os.path.join(args.preprocess_path, args.task_dataset, f'{split}_data.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

def load_data(args: argparse.Namespace):
    data_dict = {
        'image': [],
        'image_id': [],
        'question_id': [],
        'question': [],
        'answer': [],
        'full_answer': [],
        'question_type': [],
        'answer_type': [],
        'caption': [],
        'domain_id': [], # 0: normal, realistic image,
        'split': []
    }

    # Load dataset
    if args.task_dataset == 'vqa_v2':
        raw_dataset = load_dataset('HuggingFaceM4/VQAv2')

        # Load coco caption data -> attach caption to each image
        caption_dataset = load_dataset('yerevann/coco-karpathy')
        caption_data = {}
        for split in ['train', 'validation', 'test', 'restval']:
            for data in caption_dataset[split]:
                caption_data[data['cocoid']] = data['sentences'][0]

        for split in ['train', 'validation', 'testdev', 'test']:
            for data in tqdm(raw_dataset[split]):
                data_dict['image'].append(data['image'].convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                data_dict['image_id'].append(data['image_id'])
                data_dict['question_id'].append(data['question_id'])
                data_dict['question'].append(data['question'])
                data_dict['answer'].append(data['multiple_choice_answer'])
                data_dict['full_answer'].append(data['answers'])
                data_dict['question_type'].append(data['question_type'])
                data_dict['answer_type'].append(data['answer_type'])
                data_dict['domain_id'].append(0) # 0: normal, realistic image
                try:
                    data_dict['caption'].append(caption_data[data['image_id']])
                except:
                    data_dict['caption'].append("")
                if split == 'validation':
                    data_dict['split'].append('valid')
                elif split == 'test':
                    data_dict['split'].append('teststd')
                else:
                    data_dict['split'].append(split)
    elif args.task_dataset == 'uit_viic_vqa_ori':
        raw_dataset = load_dataset('HuggingFaceM4/VQAv2')
        with open('./task/captioning/UIT_VIIC_SPLIT.json', 'rb') as f:
            caption_data = json.load(f)
            caption_data = caption_data['data']

        file_path_list = []
        image_id_list = []
        split_list = []
        for each_data in caption_data:
            file_path = "train2014" if "train2014" in each_data['file_name'] else "val2014"
            fileid = int(each_data['file_name'].split('_')[-1].split('.')[0])
            file_path_list.append(os.path.join(args.data_path, 'coco_2014', file_path, each_data['file_name']))
            image_id_list.append(fileid)
            split_list.append(each_data['split'])

        for vqa_split in ['train', 'validation', 'testdev', 'test']:
            for data in tqdm(raw_dataset[vqa_split]):
                # We will use only yes/no questions
                if data['image_id'] in image_id_list and data['answer_type'] == 'yes/no':
                    image = Image.open(file_path_list[image_id_list.index(data['image_id'])]).convert('RGB')
                    image = image.resize((args.image_resize_size, args.image_resize_size))

                    data_dict['image'].append(image)
                    data_dict['image_id'].append(data['image_id'])
                    data_dict['question_id'].append(data['question_id'])
                    data_dict['question'].append(data['question'])
                    data_dict['answer'].append(data['multiple_choice_answer'])
                    data_dict['full_answer'].append(data['answers'])
                    data_dict['question_type'].append(data['question_type'])
                    data_dict['answer_type'].append(data['answer_type'])
                    data_dict['caption'].append(caption_data[image_id_list.index(data['image_id'])]['captions'][0])
                    data_dict['domain_id'].append(0) # 0: normal, realistic image
                    data_dict['split'].append(split_list[image_id_list.index(data['image_id'])])
    else:
        raise NotImplementedError(f'Not implemented dataset: {args.task_dataset}')

    return data_dict
