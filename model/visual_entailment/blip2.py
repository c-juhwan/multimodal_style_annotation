# Standard Library Modules
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import AutoProcessor, Blip2ForConditionalGeneration, GenerationConfig
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class BLIP2VEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(BLIP2VEModel, self).__init__()
        self.args = args

        if args.model_type == 'blip2':
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
        elif args.model_type == 'blip2_xxl':
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        else:
            raise ValueError(f"Invalid model type: {args.model_type}")

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

        self.gen_config = GenerationConfig(length_penalty=-1.0, num_beams=args.num_beams, max_length=args.max_seq_len,
                                           early_stopping=True, use_cache=True, do_sample=False)

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, premise, hypothesis):
        transformed_image = [self.transform_inference(img) for img in image]
        prompt_input = [f"Statement: {each_hypothesis} Determine if the statement is true, false, or undetermined based on the image." for each_hypothesis in hypothesis]
        decoder_input = [f"Short Answer:" for _ in hypothesis]
        inputs = self.processor(images=transformed_image, text=prompt_input,
                                padding="longest", return_tensors="pt")
        inputs["decoder_input_ids"] = self.processor.tokenizer(decoder_input, padding="longest",
                                                                return_tensors="pt").input_ids
        inputs["decoder_input_ids"] = inputs["decoder_input_ids"][:, :-1] # Remove EOS token to allow model to generate
        inputs = inputs.to(self.args.device)

        generated_outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                                decoder_input_ids=inputs.decoder_input_ids,
                                                pixel_values=inputs.pixel_values,
                                                length_penalty=-1.0,
                                                num_beams=self.args.num_beams,
                                                max_new_tokens=1,
                                                min_new_tokens=1,
                                                early_stopping=True)

        generated_text = self.processor.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        generated_text = [text.split("Short Answer:")[1] for text in generated_text]
        generated_text = [text.strip() for text in generated_text]

        print(prompt_input, generated_text)

        # yes = entailment = 0, unknown = neutral = 1, no = contradiction = 2
        preds = []
        # for text in generated_text:
        #     if text == "yes":
        #         preds.append(0)
        #     elif text == "no":
        #         preds.append(2)
        #     else:
        #         preds.append(1)
        for text in generated_text:
            if 'true' in text.lower() or 'yes' in text.lower():
                preds.append(0)
            elif 'un' in text.lower():
                preds.append(1)
            elif 'fal' in text.lower() or 'no' in text.lower():
                preds.append(2)
            else:
                preds.append(1)

        return preds
