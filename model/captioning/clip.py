# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel, GPT2Config, GPT2Tokenizer, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

class CLIPCaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(CLIPCaptioningModel, self).__init__()
        self.args = args

        self.encoder_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.decoder_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        encoder_config = CLIPVisionConfig().from_pretrained("openai/clip-vit-base-patch16")
        decoder_config = GPT2Config.from_pretrained("openai-community/gpt2")
        model_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k", # This is not a typo, this is tricky
            "openai-community/gpt2"
        )

        self.model.config = model_config # override the model config with CLIP encoder config
        self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self.model.config.pad_token_id = self.decoder_tokenizer.eos_token_id
        self.model.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        if self.args.model_type == 'clip_frozen':
            self.model.encoder.requires_grad_(False)

        self.transform_train = transforms.Compose([
            transforms.Resize(args.image_resize_size), # Resize to 256x256
            transforms.RandomCrop(args.image_crop_size), # Random crop to 224x224
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD) # Normalize to the mean and std of ImageNet
        ])

        self.transform_inference = transforms.Compose([
            transforms.Resize(args.image_resize_size), # Resize to 256x256
            transforms.CenterCrop(args.image_crop_size), # Center crop to 224x224
            # transforms.ToTensor(),
            # transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD) # Normalize to the mean and std of ImageNet
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
