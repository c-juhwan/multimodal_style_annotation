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
from openai import OpenAI

class GPT4VEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(GPT4VEModel, self).__init__()
        self.args = args
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY_ACL'])

        assert self.args.batch_size == 1, "Batch size must be 1 for GPT-4"

    def forward(self, image, question, label, caption=None, domain_ids=None):
        raise NotImplementedError("Inference Only")

    def generate(self, image, premise, hypothesis):
        system_prompt = self._build_default_prompt(self.args)
        user_prompt = self._attach_image_to_prompt(system_prompt, image[0], hypothesis[0])

        preds = []
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.args.gpt_model_version,
                    messages=user_prompt,
                )

                # parse the response
                generated_text = response.choices[0].message.content
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

    def _build_default_prompt(self, args):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful AI assistant that helps visual entailment tasks."
                    }
                ]
            },
        ]

        return messages

    def _attach_image_to_prompt(self, prompt, image, hypothesis):
        local_prompt = copy.deepcopy(prompt)

        local_prompt.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Does the given hypothesis entail the image? Start the response with 'True', 'False', or 'Undetermined'.\n\
    Hypothesis: {hypothesis}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._convert_to_base64(image)}",
                        "detail": "low"
                    }
                }
            ]
        })

        return local_prompt

    def _convert_to_base64(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
