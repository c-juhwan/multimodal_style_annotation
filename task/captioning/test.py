# Standard Library Modules
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
from nlgeval import NLGEval
from bert_score import BERTScorer
from BARTScore.bart_score import BARTScorer
# Pytorch Modules
import torch
torch.set_num_threads(2)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.captioning.dataset import CaptioningDataset, collate_fn
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
    dataset_test = CaptioningDataset(os.path.join(args.preprocess_path, args.test_dataset, f'test_data.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # Get model instance
    write_log(logger, "Building model")
    if args.model_type == 'vit':
        from model.captioning.vit import ViTCaptioningModel
        model = ViTCaptioningModel(args)
    elif args.model_type == 'vit_cross':
        from model.captioning.vit_cross import ViTCrossCaptioningModel
        model = ViTCrossCaptioningModel(args)
    elif args.model_type in ['clip', 'clip_frozen']:
        from model.captioning.clip import CLIPCaptioningModel
        model = CLIPCaptioningModel(args)
    elif args.model_type in ['blip', 'blip_tuned']:
        from model.captioning.blip import BLIPCaptioningModel
        model = BLIPCaptioningModel(args)
    elif args.model_type == 'blip2':
        from model.captioning.blip2 import BLIP2CaptioningModel
        model = BLIP2CaptioningModel(args)
    elif args.model_type == 'llava_mistral':
        from model.captioning.llava_mistral import LLaVAMistralCaptioningModel
        model = LLaVAMistralCaptioningModel(args)
    elif args.model_type == 'llava_llama3':
        from model.captioning.llava_llama3 import LLaVALLaMA3CaptioningModel
        model = LLaVALLaMA3CaptioningModel(args)
    elif args.model_type == 'paligemma':
        from model.captioning.paligemma import PaliGemmaCaptioningModel
        model = PaliGemmaCaptioningModel(args)
    elif args.model_type in ['gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09']:
        from model.captioning.gpt4 import GPT4CaptioningModel
        model = GPT4CaptioningModel(args)
    elif args.model_type in ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']:
        from model.captioning.claude import ClaudeCaptioningModel
        model = ClaudeCaptioningModel(args)
    elif args.model_type in ['gemini-1.0-pro-vision-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']:
        from model.captioning.gemini import GeminiCaptioningModel
        model = GeminiCaptioningModel(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model.to(device)

    # Load model weights
    if args.model_type not in ['blip_tuned', 'blip2', 'llava_mistral', 'llava_llama3', 'paligemma',
                               'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09',
                               'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307',
                               'gemini-1.0-pro-vision-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']:
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
    ref_list = []
    hyp_list = []
    result_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc="Testing")):
        # Test - Get input data from batch
        image_ids = data_dicts['image_id']
        images = data_dicts['image']
        all_captions = data_dicts['all_captions']
        domain_ids = data_dicts['domain_id']

        # Test - Forward pass
        with torch.no_grad():
            generated_captions = model.generate(images)

        # Test - calculate
        for each_pred_sentence, each_ref_sentence in zip(generated_captions, all_captions):
            each_reference = [each_ref.replace(' .', '.') for each_ref in each_ref_sentence]

            ref_list.append(each_reference) # Multiple references
            hyp_list.append(each_pred_sentence)

        for idx in range(len(image_ids)):
            result_list.append({
                'image_id': image_ids[idx],
                'domain_id': domain_ids[idx],
                'reference': all_captions[idx],
                'hypothesis': generated_captions[idx],
            })

    # Test - nlg-eval
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    NLG_Eval = NLGEval(metrics_to_omit=['CIDEr', 'SPICE', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    BERT_Eval = BERTScorer(device=args.device, model_type='bert-base-uncased')
    BART_Eval = BARTScorer(device=args.device, checkpoint='facebook/bart-large-cnn')

    # I don't know why but we need this
    _strip = lambda x: x.strip()
    ref_list2 = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    metrics_dict = NLG_Eval.compute_metrics(ref_list2, hyp_list)

    # Test - bert-score
    bert_score_P, bert_score_R, bert_score_F1, bart_score_total = 0, 0, 0, 0
    write_log(logger, "TEST - Calculating BERTScore/BARTScore metrics...")
    for each_ref, each_hyp in tqdm(zip(ref_list2[0], hyp_list), total=len(ref_list2[0]), desc=f'TEST - Calculating BERTScore&BARTScore...'):
        P, R, F1 = BERT_Eval.score([each_ref], [each_hyp])
        bert_score_P += P.item()
        bert_score_R += R.item()
        bert_score_F1 += F1.item()
        bart_score = BART_Eval.multi_ref_score([each_ref], [each_hyp], agg='max')
        bart_score_total += bart_score[0].item()
    bert_score_P /= len(ref_list2[0])
    bert_score_R /= len(ref_list2[0])
    bert_score_F1 /= len(ref_list2[0])
    bart_score_total /= len(ref_list2[0])

    # Final - End of testing
    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    write_log(logger, f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
    write_log(logger, f"TEST - Rouge_L_NLGEVAL: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")
    write_log(logger, f"TEST - BERTScore_Precision: {bert_score_P:.4f}")
    write_log(logger, f"TEST - BERTScore_Recall: {bert_score_R:.4f}")
    write_log(logger, f"TEST - BERTScore_F1: {bert_score_F1:.4f}")
    write_log(logger, f"TEST - BARTScore: {bart_score_total:.4f}")

    # Save data as json file
    save_path = os.path.join(args.result_path, args.task, args.test_dataset)
    check_path(save_path)

    result_dict = {
        'args': vars(args),
        'Bleu_1': metrics_dict['Bleu_1'],
        'Bleu_2': metrics_dict['Bleu_2'],
        'Bleu_3': metrics_dict['Bleu_3'],
        'Bleu_4': metrics_dict['Bleu_4'],
        'Bleu_avg': (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4,
        'Rouge_L_NLG': metrics_dict['ROUGE_L'],
        'Meteor': metrics_dict['METEOR'],
        'BERTScore_Precision': bert_score_P,
        'BERTScore_Recall': bert_score_R,
        'BERTScore_F1': bert_score_F1,
        'BARTScore': bart_score_total,
        'result_list': result_list,
    }
    save_name = os.path.join(save_path, f'test_result_{args.model_type}_{args.task_dataset}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    if args.use_tensorboard:
        writer.add_scalar('TEST/Bleu_1', metrics_dict['Bleu_1'], global_step=0)
        writer.add_scalar('TEST/Bleu_2', metrics_dict['Bleu_2'], global_step=0)
        writer.add_scalar('TEST/Bleu_3', metrics_dict['Bleu_3'], global_step=0)
        writer.add_scalar('TEST/Bleu_4', metrics_dict['Bleu_4'], global_step=0)
        writer.add_scalar('TEST/Bleu_avg', (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4, global_step=0)
        writer.add_scalar('TEST/Rouge_L', metrics_dict['ROUGE_L'], global_step=0)
        writer.add_scalar('TEST/Meteor', metrics_dict['METEOR'], global_step=0)
        writer.add_scalar('TEST/BERTScore_Precision', bert_score_P, global_step=0)
        writer.add_scalar('TEST/BERTScore_Recall', bert_score_R, global_step=0)
        writer.add_scalar('TEST/BERTScore_F1', bert_score_F1, global_step=0)
        writer.add_scalar('TEST/BARTScore', bart_score_total, global_step=0)

        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Training_Dataset': [args.task_dataset],
            'Test_Dataset': [args.test_dataset],
            'Model': [args.model_type],
            'Bleu_1': [metrics_dict['Bleu_1']],
            'Bleu_2': [metrics_dict['Bleu_2']],
            'Bleu_3': [metrics_dict['Bleu_3']],
            'Bleu_4': [metrics_dict['Bleu_4']],
            'Bleu_avg': [(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4],
            'Rouge_L': [metrics_dict['ROUGE_L']],
            'Meteor': [metrics_dict['METEOR']],
            'BERTScore_Precision': [bert_score_P],
            'BERTScore_Recall': [bert_score_R],
            'BERTScore_F1': [bert_score_F1],
            'BARTScore': [bart_score_total],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)

        wandb.finish()

    return metrics_dict
