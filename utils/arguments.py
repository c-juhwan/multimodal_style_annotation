# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'Multimodal_Style_Annotation'

        # Task arguments
        task_list = ['captioning', 'visual_qa', 'visual_entailment']
        self.parser.add_argument('--task', type=str, choices=task_list, default='captioning',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing', 'zs_inference', 'annotating']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['flickr8k', 'flickr30k', 'coco_karpathy',
                        'uit_viic_en_ori', 'uit_viic_en_cartoon', 'uit_viic_en_pencil', 'uit_viic_en_oil',
                        'vqa_v2', 'uit_viic_vqa_ori', 'uit_viic_vqa_cartoon', 'uit_viic_vqa_pencil', 'uit_viic_vqa_oil',
                        'snli_ve', 'snli_ve_sports_ori', 'snli_ve_sports_cartoon', 'snli_ve_sports_pencil', 'snli_ve_sports_oil']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='uit_viic_en_ori', nargs='+',
                                 help='Dataset for the task; Must be given.')
        self.parser.add_argument('--test_dataset', type=str, choices=dataset_list, default='uit_viic_en_ori',
                                 help='Dataset for testing; Must be given.')
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')

        # Path arguments - Modify these paths to fit your environment
        self.parser.add_argument('--data_path', type=str, default=f'/nas_homes/dataset/',
                                 help='Path to the dataset.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'/nas_homes/{self.user_name}/preprocessed/{self.proj_name}',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/nas_homes/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/nas_homes/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'./results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'/nas_homes/{self.user_name}/tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default=self.proj_name,
                                 help='Name of the project.')
        model_type_list = ['vit', 'vit_cross', 'clip', 'clip_frozen', 'blip', 'blip_tuned', 'blip2', 'blip2_xxl', 'vilt',
                           'llava_mistral', 'llava_llama3', 'llava_vicuna', 'llava_15_vicuna', 'paligemma', 'instructblip', # Open-source MLLM
                           'gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-1106-vision-preview', # OpenAI GPT
                           'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', # Anthropic Claude
                           'gemini-1.0-pro-vision-latest', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest'] # Google Gemini

        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='clip',
                                 help='Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the input; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=50,
                                 help='Maximum sequence length of the input; Default is 50')
        self.parser.add_argument('--dropout_rate', type=float, default=0.0,
                                 help='Dropout rate of the model; Default is 0.0')

        # Model - Domain generalization arguments
        self.parser.add_argument('--sentence_sim_scaling', type=float, default=0.29,
                                 help='Scaling factor for sentence similarity; Default is 0.29')
        self.parser.add_argument('--positive_threshold', type=float, default=0.2,
                                 help='Threshold for positive similarity; Default is 0.2')
        self.parser.add_argument('--negative_lower_bound', type=float, default=0.01,
                                 help='Lower bound for negative similarity; Default is 0.01')
        self.parser.add_argument('--negative_upper_bound', type=float, default=0.05,
                                 help='Upper bound for negative similarity; Default is 0.05')
        self.parser.add_argument('--triplet_margin', type=float, default=2.0,
                                 help='Margin for triplet loss; Default is 2.0')
        self.parser.add_argument('--task_loss_weight', type=float, default=1.0,
                                 help='Weight for the task loss; Default is 1.0')
        self.parser.add_argument('--inter_domain_loss_weight', type=float, default=0.05,
                                 help='Weight for the inter-domain loss; Default is 0.05')
        self.parser.add_argument('--intra_domain_loss_weight', type=float, default=0.05,
                                 help='Weight for the intra-domain loss; Default is 0.05')
        self.parser.add_argument('--memory_bank_size', type=int, default=128,
                                 help='Size of the memory bank; Default is 128')
        self.parser.add_argument('--momentum_coefficient', type=float, default=0.99,
                                 help='Coefficient for momentum encoder; Default is 0.99')


        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='Adam',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is None")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=3,
                                 help='Training epochs; Default is 3')
        self.parser.add_argument('--learning_rate', type=float, default=2e-5,
                                 help='Learning rate of optimizer; Default is 2e-5')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=50,
                                 help='Batch size; Default is 50')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay; Default is 0; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=0,
                                 help='Gradient clipping norm; Default is 0')
        self.parser.add_argument('--early_stopping_patience', type=int, default=10,
                                 help='Early stopping patience; No early stopping if None; Default is None')
        self.parser.add_argument('--train_valid_split', type=float, default=0.1,
                                 help='Train/Valid split ratio; Default is 0.1')
        objective_list = ['accuracy', 'f1', 'loss']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='loss',
                                 help='Objective to optimize; Default is loss')

        # Preprocessing - Image preprocessing config
        self.parser.add_argument('--image_resize_size', default=256, type=int,
                                 help='Size of resized image after preprocessing.')
        self.parser.add_argument('--image_crop_size', default=224, type=int,
                                 help='Size of cropped image after preprocessing.')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=4, type=int,
                                 help='Batch size for test; Default is 4')
        self.parser.add_argument('--num_beams', default=5, type=int,
                                 help='Number of beams for beam search; Default is 5')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--gpt_model_version', type=str, default='gpt-4o-2024-05-13')
        self.parser.add_argument('--dalle_model_version', type=str, default='dall-e-3')
        self.parser.add_argument('--error_patience', type=int, default=10,
                                 help='Error patience; Default is 10')
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=None,
                                 help='Random seed; Default is None')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=True,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default=100, type=int,
                                 help='Logging frequency; Default is 100')

    def get_args(self):
        return self.parser.parse_args()
