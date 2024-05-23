# Standard Library Modules
import os
import copy
import time
import base64
import argparse
from io import BytesIO
# Pytorch Modules
import torch
import torch.nn as nn
# 3rd-Party Modules
from tqdm.auto import tqdm
import google.generativeai as genai
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

safety_settings=[
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class GeminiCaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(GeminiCaptioningModel, self).__init__()
        self.args = args
        self.model = genai.GenerativeModel(self.args.gpt_model_version, safety_settings=safety_settings)

        assert self.args.batch_size == 1, "Batch size must be 1 for Gemini"

    def forward(self, image, caption, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image):
        # system_prompt = self._build_default_prompt(self.args)
        # user_prompt = self._attach_image_to_prompt(system_prompt, image[0])

        while True:
            try:
                response = self.model.generate_content([
                    "You are a helpful AI assistant that helps people generate captions for their images. Your output should be a single sentence that describes the image. Do not generate any inappropriate or accompanying text.",
                    "Please generate a caption for this image. Please generate the result in the form of 'Caption: <your caption here>'",
                    image[0]
                ])

                # parse the response
                generated_text = response.text
                generated_text = generated_text.split("Caption: ")[-1].strip()
                break
            except Exception as e:
                # If failed to get a response or failed to parse the response, retry
                tqdm.write(f"Error: {e}")
                time.sleep(0.5)
                continue

        return [generated_text]
