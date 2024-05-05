# Standard Library Modules
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import logging
import argparse
# 3rd-party Modules
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
torch.set_num_threads(2)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.visual_qa.dataset import VQADataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def testing(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset...")
    if args.test_dataset == 'uit_viic_vqa_ori':
        dataset_test = VQADataset(os.path.join(args.preprocess_path, args.test_dataset, f'test_data.pkl'))
    elif args.test_dataset == 'vqa_v2':
        dataset_test = VQADataset(os.path.join(args.preprocess_path, args.test_dataset, f'testdev_data.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # Get model instance
    write_log(logger, "Building model")
    if args.model_type == 'vit':
        from model.visual_qa.vit import ViTVQAModel
        model = ViTVQAModel(args)
    elif args.model_type == 'vit_cross':
        from model.visual_qa.vit_cross import ViTCrossVQAModel
        model = ViTCrossVQAModel(args)
    elif args.model_type in ['clip', 'clip_frozen']:
        from model.visual_qa.clip import CLIPVQAModel
        model = CLIPVQAModel(args)
    elif args.model_type in ['blip', 'blip_tuned']:
        from model.visual_qa.blip import BLIPVQAModel
        model = BLIPVQAModel(args)
    elif args.model_type == 'blip2':
        from model.visual_qa.blip2 import BLIP2VQAModel
        model = BLIP2VQAModel(args)
    elif args.model_type == 'vilt':
        from model.visual_qa.vilt import ViltVQAModel
        model = ViltVQAModel(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.to(device)

    # Load model weights
    if args.model_type not in ['blip_tuned', 'vilt', 'blip2']:
        write_log(logger, "Loading model weights")
        load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                    f'{args.model_type}_final_model.pt')
        model = model.to('cpu')
        checkpoint = torch.load(load_model_name, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        write_log(logger, f"Loaded model weights from {load_model_name}")
        del checkpoint # Delete checkpoint to free memory

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Initialize wandb
    if args.use_wandb:
        import wandb # Only import wandb when it is used
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TEST",
                             f"Training_Dataset: {args.task_dataset}",
                             f"Test_Dataset: {args.test_dataset}",
                             f"Model: {args.model_type}"])

    # Test - Start evaluation
    model = model.eval()
    result_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc="Testing")):
        # Test - Get input data from batch
        image_ids = data_dicts['image_id']
        question_ids = data_dicts['question_id']
        question_types = data_dicts['question_type']
        answer_types = data_dicts['answer_type']
        images = data_dicts['image']
        questions = data_dicts['question']
        gold_answers = data_dicts['answer']
        full_answers = data_dicts['full_answer']

        # Test - Forward pass
        with torch.no_grad():
            generated_answers = model.generate(images, questions)

        # Test - Store results
        for each_answer, each_gold_answer, each_full_answer, image_id, question_id, question_type, answer_type \
        in zip(generated_answers, gold_answers, full_answers, image_ids, question_ids, question_types, answer_types):
            result_list.append({
                'generated_answer': each_answer,
                'gold_answer': each_gold_answer,
                'full_answer': each_full_answer,
                'image_id': image_id,
                'question_id': question_id,
                'question_type': question_type,
                'answer_type': answer_type
            })

    # Test - Calculate accuracy
    for each_result in result_list:
        pred = process_text(each_result['generated_answer'])
        human_answers = [each_answer['answer'] for each_answer in each_result['full_answer']]
        ground_truths = [process_text(each_answer) for each_answer in human_answers]

        matching_answer = []
        for each_ground_truth in ground_truths:
            if each_ground_truth == pred:
                matching_answer.append(each_ground_truth)

        accuracy = min(1, float(len(matching_answer)) / 3)
        result_list[result_list.index(each_result)]['accuracy'] = accuracy
    average_accuracy = sum([each_result['accuracy'] for each_result in result_list]) / len(result_list)

    # Final - End of testing
    write_log(logger, f"TEST - Average accuracy: {average_accuracy:.4f}")

    # Save data as json file
    save_path = os.path.join(args.result_path, args.task, args.test_dataset)
    check_path(save_path)

    result_dict = {
        'args': vars(args),
        'average_accuracy': average_accuracy,
        'result_list': result_list
    }
    save_name = os.path.join(save_path, f'test_result_{args.model_type}_{args.task_dataset}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    if args.use_tensorboard:
        writer.add_scalar('Test/Average_Accuracy', average_accuracy, 0)
        writer.close()

    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Training_Dataset': [args.task_dataset],
            'Test_Dataset': [args.test_dataset],
            'Model': [args.model_type],
            'Average_Accuracy': [average_accuracy]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return average_accuracy

def process_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.strip()
    text = process_punct(text)
    text = process_digit_article(text)

    return text

def process_punct(text):
    period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile("(\d)(\,)(\d)")
    punct_list = [';', r"/", '[', ']', '"', '{', '}',
				  '(', ')', '=', '+', '\\', '_', '-',
				  '>', '<', '@', '`', ',', '?', '!']

    output = text

    for p in punct_list:
        if (p + ' ' in text or ' ' + p in text) or (re.search(comma_strip, text) != None):
            output = output.replace(p, '')
        else:
            output = output.replace(p, ' ')
    output = period_strip.sub("", output, re.UNICODE)

    return output

def process_digit_article(text):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                    "youll": "you'll", "youre": "you're", "youve": "you've"}
    manual_digit_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2',
        'three': '3', 'four': '4', 'five': '5', 'six': '6',
        'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    articles = ['a', 'an', 'the']

    output = []
    temp_text = text.lower().split()

    for word in temp_text:
        word = manual_digit_map.setdefault(word, word)
        if word not in articles:
            output.append(word)
        else:
            pass

    for i, word in enumerate(output):
        if word in contractions:
            output[i] = contractions[word]

    output = ' '.join(output)
    return output
