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
from model.visual_entailment.dataset import VEDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path, list_to_str_wandb

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
    dataset_test = VEDataset(os.path.join(args.preprocess_path, args.test_dataset, f'test_data.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # Get model instance
    write_log(logger, "Building model")
    if args.model_type == 'vit':
        from model.visual_entailment.vit import ViTVEModel
        model = ViTVEModel(args)
    elif args.model_type == 'vit_cross':
        from model.visual_entailment.vit_cross import ViTCrossVEModel
        model = ViTCrossVEModel(args)
    elif args.model_type in ['clip', 'clip_frozen']:
        from model.visual_entailment.clip import CLIPVEModel
        model = CLIPVEModel(args)
    elif args.model_type in ['blip', 'blip_tuned']:
        from model.visual_entailment.blip import BLIPVEModel
        model = BLIPVEModel(args)
    elif args.model_type in ['blip2', 'blip2_xxl']:
        from model.visual_entailment.blip2 import BLIP2VEModel
        model = BLIP2VEModel(args)
    elif args.model_type == 'instructblip':
        from model.captioning.instructblip import InstructBlipCaptioningModel
        model = InstructBlipCaptioningModel(args)
    elif args.model_type == 'llava_15_vicuna':
        from model.visual_entailment.llava_15_vicuna import LLaVA15VicunaVEModel
        model = LLaVA15VicunaVEModel(args)
    elif args.model_type == 'llava_mistral':
        from model.visual_entailment.llava_mistral import LLaVAMistralVEModel
        model = LLaVAMistralVEModel(args)
    elif args.model_type == 'llava_vicuna':
        from model.visual_entailment.llava_vicuna import LLaVAVicunaVEModel
        model = LLaVAVicunaVEModel(args)
    elif args.model_type == 'llava_llama3':
        from model.visual_entailment.llava_llama3 import LLaVALLaMA3VEModel
        model = LLaVALLaMA3VEModel(args)
    elif args.model_type == 'paligemma':
        from model.visual_entailment.paligemma import PaliGemmaVEModel
        model = PaliGemmaVEModel(args)
    elif args.model_type in ['gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-1106-vision-preview']:
        from model.visual_entailment.gpt4 import GPT4VEModel
        model = GPT4VEModel(args)
    elif args.model_type in ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']:
        from model.visual_entailment.claude import ClaudeVEModel
        model = ClaudeVEModel(args)
    elif args.model_type in ['gemini-1.0-pro-vision-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']:
        from model.visual_entailment.gemini import GeminiVEModel
        model = GeminiVEModel(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.to(device)

    # Load model weights
    if args.model_type not in ['blip_tuned', 'blip2', 'blip2_xxl',  'paligemma', 'instructblip',
                               'llava_15_vicuna', 'llava_vicuna',  'llava_mistral', 'llava_llama3',
                               'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-1106-vision-preview',
                               'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307',
                               'gemini-1.0-pro-vision-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']:
        write_log(logger, "Loading model weights")
        load_model_name = os.path.join(args.model_path, args.task, list_to_str_wandb(args.task_dataset),
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
                             f"Training_Dataset: {list_to_str_wandb(args.task_dataset)}",
                             f"Test_Dataset: {args.test_dataset}",
                             f"Model: {args.model_type}"])

    # Test - Start evaluation
    model = model.eval()
    result_list = []

    test_acc = 0
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc="Testing")):
        # Test - Get input data from batch
        image_ids = data_dicts['image_id']
        images = data_dicts['image']
        premise = data_dicts['premise']
        hypothesis = data_dicts['hypothesis']
        labels = data_dicts['label']

        # Test - Forward pass
        with torch.no_grad():
            generated_answers = model.generate(images, premise, hypothesis)

        # Test - Store results
        for each_answer, each_label, each_premise, each_hypothesis, image_id \
        in zip(generated_answers, labels, premise, hypothesis, image_ids):
            result_list.append({
                'image_id': image_id,
                'premise': each_premise,
                'hypothesis': each_hypothesis,
                'generated_answer': each_answer,
                'label': each_label,
                'correct': each_answer == each_label
            })

        test_acc += sum([1 if each['correct'] else 0 for each in result_list[-len(labels):]])
    test_acc /= len(result_list)

    # Final - End of testing
    write_log(logger, f"TEST - Average accuracy: {test_acc:.4f}")

    # Save data as json file
    save_path = os.path.join(args.result_path, args.task, args.test_dataset)
    check_path(save_path)

    result_dict = {
        'args': vars(args),
        'test_accuracy': test_acc,
        'result_list': result_list
    }
    save_name = os.path.join(save_path, f'test_result_{args.model_type}_{list_to_str_wandb(args.task_dataset)}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    if args.use_tensorboard:
        writer.add_scalar('Test/Average_Accuracy', test_acc, 0)
        writer.close()

    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Training_Dataset': [list_to_str_wandb(args.task_dataset)],
            'Test_Dataset': [args.test_dataset],
            'Model': [args.model_type],
            'Average_Accuracy': [test_acc]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return test_acc
