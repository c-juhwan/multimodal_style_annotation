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

class ClaudeVEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClaudeVEModel, self).__init__()
        self.args = args
        self.client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        assert self.args.batch_size == 1, "Batch size must be 1 for Claude"

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, premise, hypothesis):
        user_prompt = self._attach_image_to_prompt(image[0], hypothesis[0])

        preds=[]
        while True:
            try:
                response = self.client.messages.create(
                    model=self.args.gpt_model_version,
                    system="You are a helpful AI assistant that helps visual entailment tasks.",
                    messages=user_prompt,
                    max_tokens=10,
                )

                # parse the response
                generated_text = response.content[0].text
                if generated_text.strip().lower().startswith("true"):
                    preds.append(0)
                elif generated_text.strip().lower().startswith("false"):
                    preds.append(2)
                else:
                    preds.append(1)
                break
            except Exception as e:
                # If failed to get a response or failed to parse the response, retry
                tqdm.write(f"Error: {e}")
                time.sleep(0.5)
                continue

        return preds

    def _attach_image_to_prompt(self, image, hypothesis):
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
                        "text": f"Does the given hypothesis entail the image? Start the response with 'True', 'False', or 'Undetermined'.\n\
Hypothesis: {hypothesis}"
                    },
                ]
            },
        ]

        return messages

    def _convert_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
