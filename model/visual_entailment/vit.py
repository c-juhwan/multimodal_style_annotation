# Standard Library Modules
import copy
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import ViTImageProcessor, ViTModel
from transformers import BertTokenizer, BertModel
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class ViTVEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ViTVEModel, self).__init__()
        self.args = args

        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

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

        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size + self.text_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # Visual entailment: 3-class classification (yes, no, unknown)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, image, premise, hypothesis, label, domain_ids=None):
        # Encode image
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.image_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        text_encoder_input = self.text_tokenizer(hypothesis, padding="longest", return_tensors="pt")

        pixel_values = pixel_values.to(self.args.device)
        text_encoder_input = {key: val.to(self.args.device) for key, val in text_encoder_input.items()}

        image_encoder_output = self.image_encoder(pixel_values=pixel_values)["last_hidden_state"]
        text_encoder_output = self.text_encoder(input_ids=text_encoder_input["input_ids"],
                                                attention_mask=text_encoder_input["attention_mask"],
                                                token_type_ids=text_encoder_input["token_type_ids"])["last_hidden_state"][:, 0, :] # CLS token
        image_encoder_output = image_encoder_output[:, 0, :] # CLS token

        combined_output = torch.cat((image_encoder_output, text_encoder_output), dim=1)
        logits = self.classifier(combined_output)

        target_tensor = torch.tensor(label, dtype=torch.long, device=self.args.device)
        loss = self.loss_func(logits, target_tensor)

        output = {
            'logits': logits,
            'loss': loss
        }
        return output

    def generate(self, image, premise, hypothesis):
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.image_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        text_encoder_input = self.text_tokenizer(hypothesis, padding="longest", return_tensors="pt")

        pixel_values = pixel_values.to(self.args.device)
        text_encoder_input = {key: val.to(self.args.device) for key, val in text_encoder_input.items()}

        image_encoder_output = self.image_encoder(pixel_values=pixel_values)["last_hidden_state"]
        text_encoder_output = self.text_encoder(input_ids=text_encoder_input["input_ids"],
                                                attention_mask=text_encoder_input["attention_mask"],
                                                token_type_ids=text_encoder_input["token_type_ids"])["last_hidden_state"][:, 0, :] # CLS token
        image_encoder_output = image_encoder_output[:, 0, :] # CLS token

        combined_output = torch.cat((image_encoder_output, text_encoder_output), dim=1)
        logits = self.classifier(combined_output)
        preds = logits.argmax(-1).tolist()

        return preds
