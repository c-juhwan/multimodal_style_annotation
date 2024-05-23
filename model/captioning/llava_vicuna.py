# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class LLaVAVicunaCaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(LLaVAVicunaCaptioningModel, self).__init__()
        self.args = args

        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")

        self.transform_train = transforms.Compose([
            transforms.Resize(args.image_resize_size), # Resize to 256x256
            transforms.RandomCrop(args.image_crop_size), # Random crop to 224x224
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize(IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD) # Normalize to the mean and std of ImageNet
        ])

        self.transform_inference = transforms.Compose([
            transforms.Resize(args.image_resize_size), # Resize to 256x256
            transforms.CenterCrop(args.image_crop_size), # Center crop to 224x224
            # transforms.ToTensor(),
            # transforms.Normalize(IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD) # Normalize to the mean and std of ImageNet
        ])

    def forward(self, image, caption, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image):
        transformed_image = [self.transform_inference(img) for img in image]
        prompt_input = ["USER: <image>\nProvide a detailed description of the given image in one sentence. ASSISTANT:" for _ in range(len(image))]

        inputs = self.processor(prompt_input, transformed_image, return_tensors="pt").to(self.args.device)

        generated_outputs = self.model.generate(**inputs,
                                                num_beams=self.args.num_beams,
                                                max_new_tokens=self.args.max_seq_len, early_stopping=True)

        generated_text = self.processor.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_text = [text.split("ASSISTANT:")[1] for text in generated_text]
        generated_text = [text.strip() for text in generated_text]

        return generated_text
