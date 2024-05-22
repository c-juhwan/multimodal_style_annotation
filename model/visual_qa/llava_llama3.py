# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class LLaVALLaMA3VQAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(LLaVALLaMA3VQAModel, self).__init__()
        self.args = args

        self.processor = AutoProcessor.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers")
        self.model = LlavaForConditionalGeneration.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers")

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

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, question):
        transformed_image = [self.transform_inference(img) for img in image]
        prompt_input = [f"<|start_header_id|>user<|end_header_id|>\n\n<image>\nQuestion: based on the image, {each_question}\nAnswer with yes or no.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for each_question in question]

        inputs = self.processor(prompt_input, transformed_image, return_tensors="pt").to(self.args.device)


        generated_outputs = self.model.generate(**inputs,
                                                max_new_tokens=1,
                                                min_new_tokens=1,
                                                early_stopping=True)

        generated_text = self.processor.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_text = [text.split("\n\n")[-1] for text in generated_text]
        generated_text = [text.split("<|eot_id|>")[0] for text in generated_text]
        generated_text = [text.strip() for text in generated_text]

        preds = []
        for text in generated_text:
            if "yes" in text.lower():
                preds.append("yes")
            elif "no" in text.lower():
                preds.append("no")

        return preds
