# Standard Library Modules
import re
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipConfig
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
# git clone https://github.com/salesforce/BLIP.git
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from BLIP.models.blip import blip_decoder

class BLIPCaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(BLIPCaptioningModel, self).__init__()
        self.args = args

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        if self.args.model_type == 'blip_tuned':
            # In this case, load pretrained model (already fine-tuned)
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
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

    def forward(self, image, caption, domain_ids=None):
        # Encode image
        # image = torch.stack([self.transform(img) for img in image])
        transformed_image = [self.transform_train(img) for img in image]

        # Bypass the error
        encoding_image_processor = self.processor.image_processor(images=transformed_image, return_tensors="pt",
                                                                  do_resize=True, do_rescale=True, do_normalize=True)
        text_encoding = self.processor.tokenizer(text=caption, padding='max_length', truncation=True,
                                                 max_length=self.args.max_seq_len, return_tensors="pt")
        encoding_image_processor.update(text_encoding)
        inputs = encoding_image_processor.to(self.args.device)
        inputs["labels"] = inputs["input_ids"].clone() # Assign the input_ids to the labels

        outputs = self.model(**inputs)
        return outputs

    def generate(self, image):
        transformed_image = [self.transform_inference(img) for img in image]
        inputs = self.processor.image_processor(images=transformed_image, return_tensors="pt",
                                                do_resize=True, do_rescale=True, do_normalize=True).to(self.args.device)

        generated_outputs = self.model.generate(**inputs, num_beams=self.args.num_beams,
                                                #pad_token_id=self.processor.tokenizer.eos_token_id,
                                                max_new_tokens=self.args.max_seq_len, early_stopping=True)

        generated_text = self.processor.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_text = [text.strip() for text in generated_text]

        return generated_text

def convert_blip_checkpoint():
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    config = BlipConfig.from_pretrained("Salesforce/blip-image-captioning-base")
    hf_model = BlipForConditionalGeneration(config).eval()

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

def rename_key(key):
    if "visual_encoder" in key:
        key = re.sub("visual_encoder*", "vision_model.encoder", key)
    if "blocks" in key:
        key = re.sub(r"blocks", "layers", key)
    if "attn" in key:
        key = re.sub(r"attn", "self_attn", key)
    if "norm1" in key:
        key = re.sub(r"norm1", "layer_norm1", key)
    if "norm2" in key:
        key = re.sub(r"norm2", "layer_norm2", key)
    if "encoder.norm" in key:
        key = re.sub(r"encoder.norm", "post_layernorm", key)
    if "encoder.patch_embed.proj" in key:
        key = re.sub(r"encoder.patch_embed.proj", "embeddings.patch_embedding", key)

    if "encoder.pos_embed" in key:
        key = re.sub(r"encoder.pos_embed", "embeddings.position_embedding", key)
    if "encoder.cls_token" in key:
        key = re.sub(r"encoder.cls_token", "embeddings.class_embedding", key)

    if "self_attn" in key:
        key = re.sub(r"self_attn.proj", "self_attn.projection", key)

    return key
