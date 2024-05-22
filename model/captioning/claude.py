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
import anthropic

class ClaudeCaptioningModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClaudeCaptioningModel, self).__init__()
        self.args = args
        self.client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        assert self.args.batch_size == 1, "Batch size must be 1 for Claude"

    def forward(self, image, caption, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image):
        user_prompt = self._attach_image_to_prompt(image[0])

        while True:
            try:
                response = self.client.messages.create(
                    model=self.args.gpt_model_version,
                    system="You are a helpful AI assistant that helps people generate captions for their images. Your output should be a single sentence that describes the image. Do not generate any inappropriate or accompanying text.",
                    messages=user_prompt,
                    max_tokens=100,
                )

                # parse the response
                generated_text = response.content[0].text
                generated_text = generated_text.split("Caption: ")[-1].strip()
                break
            except Exception as e:
                # If failed to get a response or failed to parse the response, retry
                tqdm.write(f"Error: {e}")
                time.sleep(0.5)
                continue

        return [generated_text]

    def _attach_image_to_prompt(self, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._convert_to_base64(image)
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please generate a caption for this image. Please generate the result in the form of 'Caption: <your caption here>'"
                    },
                ]
            },
        ]

        return messages

    def _convert_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
