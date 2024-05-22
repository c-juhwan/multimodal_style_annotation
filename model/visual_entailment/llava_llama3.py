# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class LLaVALLaMA3VEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(LLaVALLaMA3VEModel, self).__init__()
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

    def generate(self, image, premise, hypothesis):
        transformed_image = [self.transform_inference(img) for img in image]
        prompt_input = [f"<|start_header_id|>user<|end_header_id|>\n\n<image>\nStatement: {each_hypothesis}\nDetermine if the statement is true, false, or undetermined based on the image. Answer with true, false, or undetermined.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for each_hypothesis in hypothesis]

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
            if "true" in text.lower():
                preds.append(0)
            elif "false" in text.lower():
                preds.append(2)
            else:
                preds.append(1)

        return preds
