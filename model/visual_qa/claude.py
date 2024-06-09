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

class ClaudeVQAModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClaudeVQAModel, self).__init__()
        self.args = args
        self.client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        assert self.args.batch_size == 1, "Batch size must be 1 for Claude"

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, question):
        user_prompt = self._attach_image_to_prompt(image[0], question[0])

        preds = []
        error_count = 0
        while True:
            try:
                response = self.client.messages.create(
                    model=self.args.gpt_model_version,
                    system="You are a helpful AI assistant that helps visual question answering tasks. You must start the answer in the form of 'Yes' or 'No'.",
                    messages=user_prompt,
                    max_tokens=10,
                )

                # parse the response
                generated_text = response.content[0].text
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

    def _attach_image_to_prompt(self, image, question):
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
                        "text": f"Please answer the question below based on the given image. Start the response with 'Yes' or 'No'.\n\
Question: {question}"
                    },
                ]
            },
        ]

        return messages

    def _convert_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
