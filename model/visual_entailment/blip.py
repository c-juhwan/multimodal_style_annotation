# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration, BlipConfig
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
# git clone https://github.com/salesforce/BLIP.git
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from BLIP.models.blip import blip_decoder
from model.captioning.blip import rename_key

class BLIPVEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(BLIPVEModel, self).__init__()
        self.args = args

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        if self.args.model_type == 'blip_tuned':
            # In this case, load pretrained model (already fine-tuned)
            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        else:
            # In this case, load the model from the checkpoint
            self.model = convert_blip_checkpoint()

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
            nn.Linear(self.model.text_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # Visual entailment: 3-class classification (yes, no, unknown)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, image, premise, hypothesis, label, domain_ids=None):
        # Encode image
        # image = torch.stack([self.transform(img) for img in image])
        transformed_image = [self.transform_train(img) for img in image]
        inputs = self.processor(images=transformed_image, text=hypothesis,
                                padding="longest", return_tensors="pt").to(self.args.device)

        vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
        image_embeds = vision_outputs[0] # Last hidden state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        question_embeds = self.model.text_encoder(input_ids=inputs.input_ids,
                                            attention_mask=inputs.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_attention_mask)
        question_embeds = question_embeds[0][:, 0, :] # CLS token of last layer

        logits = self.classifier(question_embeds)
        target_tensor = torch.tensor(label, dtype=torch.long, device=self.args.device)
        loss = self.loss_func(logits, target_tensor)

        outputs = {
            'logits': logits,
            'loss': loss
        }
        return outputs

    def generate(self, image, premise, hypothesis):
        transformed_image = [self.transform_inference(img) for img in image]
        inputs = self.processor(images=transformed_image, text=hypothesis,
                                padding="longest", return_tensors="pt").to(self.args.device)

        vision_outputs = self.model.vision_model(pixel_values=inputs.pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        question_embeds = self.model.text_encoder(input_ids=inputs.input_ids,
                                            attention_mask=inputs.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_attention_mask)
        question_embeds = question_embeds[0][:, 0, :] # CLS token of last layer

        logits = self.classifier(question_embeds)
        preds = logits.argmax(-1).tolist()
        return preds

def convert_blip_checkpoint():
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    config = BlipConfig.from_pretrained("Salesforce/blip-vqa-base")
    hf_model = BlipForQuestionAnswering(config).eval()

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
    pt_model = blip_decoder(pretrained=model_url, image_size=384, vit="base")
    pt_model = pt_model.eval()

    modified_state_dict = pt_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    hf_model.load_state_dict(modified_state_dict, strict=False)

    return hf_model
