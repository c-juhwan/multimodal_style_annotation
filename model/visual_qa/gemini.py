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
from PIL import PngImagePlugin

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

class GeminiVQAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(GeminiVQAModel, self).__init__()
        self.args = args
        self.model = genai.GenerativeModel(self.args.gpt_model_version, safety_settings=safety_settings)

        assert self.args.batch_size == 1, "Batch size must be 1 for Gemini"

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, question):
        # system_prompt = self._build_default_prompt(self.args)
        # user_prompt = self._attach_image_to_prompt(system_prompt, image[0])

        preds = []
        error_count = 0
        while True:
            try:
                response = self.model.generate_content([
                    "You are a helpful AI assistant that helpsYou are a helpful AI assistant that helps visual question answering tasks.",
                    f"Please answer the question below based on the given image. Start the response with 'Yes' or 'No'.\nQuestion: {question}",
                    image[0]
                ])

                # parse the response
                generated_text = response.text
                tqdm.write(f"Generated Text: {generated_text}")
                if generated_text.strip().lower().startswith("yes"):
                    preds.append("yes")
                elif generated_text.strip().lower().startswith("no"):
                    preds.append("no")
                else:
                    error_count += 1
                    if error_count > 5:
                        preds.append("error")
                        break
                    continue
                break
            except Exception as e:
                # If failed to get a response or failed to parse the response, retry
                tqdm.write(f"Error: {e}")
                time.sleep(0.5)
                continue

        return preds
