# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class ViltVQAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ViltVQAModel, self).__init__()
        self.args = args

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=self.args.max_seq_len)
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        pred_idx = outputs.logits.argmax(-1).tolist()
        pred_answer = [self.model.config.id2label[each_pred] for each_pred in pred_idx]

        return pred_answer
