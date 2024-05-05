# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class BLIP2CaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(BLIP2CaptioningModel, self).__init__()
        self.args = args

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

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
        # transformed_image = [self.transform_train(img) for img in image]

        # # Bypass the error
        # encoding_image_processor = self.processor.image_processor(images=transformed_image, return_tensors="pt",
        #                                                           do_resize=True, do_rescale=True, do_normalize=True)
        # text_encoding = self.processor.tokenizer(text=caption, padding='max_length', truncation=True,
        #                                          max_length=self.args.max_seq_len, return_tensors="pt")
        # encoding_image_processor.update(text_encoding)
        # inputs = encoding_image_processor.to(self.args.device)
        # inputs["labels"] = inputs["input_ids"].clone() # Assign the input_ids to the labels

        # outputs = self.model(**inputs)
        # return outputs
        raise NotImplementedError("Inference Only")


    def generate(self, image):
        transformed_image = [self.transform_inference(img) for img in image]
        prompt_input = ["Image description: " for _ in range(len(image))]

        inputs = self.processor(images=transformed_image, text=prompt_input,
                                padding="longest", return_tensors="pt").to(self.args.device)

        generated_outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                                pixel_values=inputs.pixel_values,
                                                num_beams=self.args.num_beams,
                                                max_new_tokens=self.args.max_seq_len, early_stopping=True)

        generated_text = self.processor.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_text = [text.strip() for text in generated_text]

        return generated_text
