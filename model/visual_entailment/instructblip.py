# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, InstructBlipForConditionalGeneration
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class InstructBlipVEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(InstructBlipVEModel, self).__init__()
        self.args = args

        self.processor = AutoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")

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

    def generate(self, image, premise, hypothesis):
        transformed_image = [self.transform_inference(img) for img in image]
        prompt_input = [f"Statement: {each_hypothesis}\nDetermine if the statement is true, false, or undetermined based on the image. Answer with true, false, or undetermined." for each_hypothesis in hypothesis]

        inputs = self.processor(images=transformed_image, text=prompt_input, return_tensors="pt").to(self.args.device)

        generated_outputs = self.model.generate(**inputs,
                                                max_new_tokens=1,
                                                min_new_tokens=1,
                                                early_stopping=True)

        generated_text = self.processor.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_text = [text.split("ASSISTANT:")[1] for text in generated_text]
        generated_text = [text.strip() for text in generated_text]

        preds = []
        for text in generated_text:
            if "true" in text.lower():
                preds.append(0)
            elif "false" in text.lower():
                preds.append(2)
            else:
                preds.append(1)

        return preds
