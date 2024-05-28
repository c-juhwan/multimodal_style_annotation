# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers import BertTokenizer, BertModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

class CLIPVQAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(CLIPVQAModel, self).__init__()
        self.args = args

        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        if self.args.model_type == 'clip_frozen':
            self.image_encoder.requires_grad_(False)

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
            nn.Linear(512, 2) # We will only deal with yes/no questions
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, image, question, label, caption=None, domain_ids=None):
        # Encode image
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.image_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        input_prompt = [f"{each_question}" for each_question in question]
        target = [0 if each_label == "no" else 1 for each_label in label]
        text_encoder_input = self.text_tokenizer(input_prompt, padding="longest", return_tensors="pt")

        pixel_values = pixel_values.to(self.args.device)
        text_encoder_input = {key: val.to(self.args.device) for key, val in text_encoder_input.items()}

        image_encoder_output = self.image_encoder(pixel_values=pixel_values)["last_hidden_state"]
        text_encoder_output = self.text_encoder(input_ids=text_encoder_input["input_ids"],
                                                attention_mask=text_encoder_input["attention_mask"],
                                                token_type_ids=text_encoder_input["token_type_ids"])["last_hidden_state"][:, 0, :] # CLS token
        image_encoder_output = image_encoder_output[:, 0, :] # CLS token

        combined_output = torch.cat((image_encoder_output, text_encoder_output), dim=1)
        logits = self.classifier(combined_output)

        target_tensor = torch.tensor(target, dtype=torch.long, device=self.args.device)
        loss = self.loss_func(logits, target_tensor)
        accuracy = (logits.argmax(-1) == target_tensor).sum().item() / len(target)

        output = {
            'logits': logits,
            'loss': loss,
            'accuracy': accuracy
        }
        return output

    def generate(self, image, question):
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.image_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        input_prompt = [f"{each_question}" for each_question in question]
        text_encoder_input = self.text_tokenizer(input_prompt, padding="longest", return_tensors="pt")

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
        preds = ["no" if each_pred == 0 else "yes" for each_pred in preds]

        return preds
