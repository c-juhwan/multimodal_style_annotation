# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import ViTImageProcessor, GPT2Tokenizer, VisionEncoderDecoderModel
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class ViTCaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ViTCaptioningModel, self).__init__()
        self.args = args

        self.encoder_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.decoder_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token # Set the pad token to be the end of sequence token

        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k",
            "openai-community/gpt2"
        )
        self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self.model.config.pad_token_id = self.decoder_tokenizer.eos_token_id

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
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.encoder_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        caption_id = self.decoder_tokenizer(caption, padding='max_length',
                                            truncation=True, max_length=self.args.max_seq_len,
                                            return_tensors="pt").input_ids
        pixel_values = pixel_values.to(self.args.device)
        caption_id = caption_id.to(self.args.device)

        outputs = self.model(pixel_values=pixel_values, labels=caption_id)
        return outputs

    def generate(self, image):
        transformed_image = [self.transform_inference(img) for img in image]
        pixel_values = self.encoder_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        pixel_values = pixel_values.to(self.args.device)

        generated_ids = self.model.generate(pixel_values=pixel_values, num_beams=self.args.num_beams,
                                            pad_token_id=self.decoder_tokenizer.eos_token_id,
                                            max_new_tokens=self.args.max_seq_len, early_stopping=True)
        generated_text = self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [text.strip() for text in generated_text]

        return generated_text
