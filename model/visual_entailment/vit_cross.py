# Standard Library Modules
import copy
import argparse
# 3rd-party Modules
import spacy # python -m spacy download en_core_web_sm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# Pytorch Modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# Huggingface Module
from transformers import ViTImageProcessor, ViTModel
from transformers import BertTokenizer, BertModel
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD

class ViTCrossVEModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ViTCrossVEModel, self).__init__()
        self.args = args

        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.momentum_encoder = copy.deepcopy(self.image_encoder) # Momentum encoder for domain generalization
        self.momentum_encoder.requires_grad_(False) # No need to update the momentum encoder

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

        self.pos_tagger = spacy.load("en_core_web_sm")
        self.sentence_bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.memory_bank = {
            'memory_cls': None,
            'memory_domain_id': None,
            'caption': None,
            'caption_embedding': None,
            'caption_pos_tag': None
        }

        self.anchor_caption = None
        self.anchor_pos_tag = None
        self.anchor_embedding = None

        self.classifier = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size + self.text_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # Visual entailment: 3-class classification (yes, no, unknown)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, image, premise, hypothesis, label, domain_ids=None):
        # Encode image
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.image_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        text_encoder_input = self.text_tokenizer(hypothesis, padding="longest", return_tensors="pt")

        pixel_values = pixel_values.to(self.args.device)
        text_encoder_input = {key: val.to(self.args.device) for key, val in text_encoder_input.items()}

        # Update momentum encoder
        self._update_momentum_encoder()

        # Initial memory bank update
        encoder_cls = self._get_momentum_encoding(pixel_values)

        target_pos_tag_list = [] # Placeholder for the anchor caption
        visual_word_similarity = [self._calc_visual_word_similarity(premise[0], premise[i])
                                  for i in range(1, len(premise))]
        target_pos_tag_list.extend([similarity[1] for similarity in visual_word_similarity])
        visual_word_similarity = [similarity[0] for similarity in visual_word_similarity]

        target_embedding_list = [] # Placeholder for the anchor caption
        sentence_similarity = [self._calc_sentence_similarity(premise[0], premise[i])
                              for i in range(1, len(premise))]
        target_embedding_list.extend([similarity[1] for similarity in sentence_similarity])
        sentence_similarity = [similarity[0] for similarity in sentence_similarity]

        # Update memory bank
        pos_idx, neg_idx = self._get_pos_neg_idx(visual_word_similarity, sentence_similarity)
        self._update_memory_bank(encoder_cls[pos_idx],
                                 [domain_ids[i] for i in pos_idx],
                                 [premise[i] for i in pos_idx],
                                 [target_embedding_list[i] for i in pos_idx],
                                 [target_pos_tag_list[i] for i in pos_idx])
        self._update_memory_bank(encoder_cls[neg_idx],
                                 [domain_ids[i] for i in neg_idx],
                                 [premise[i] for i in neg_idx],
                                 [target_embedding_list[i] for i in neg_idx],
                                 [target_pos_tag_list[i] for i in neg_idx])

        # Inter-domain loss
        inter_domain_loss = self._calc_inter_domain_loss(encoder_cls[0], premise[0])
        # Intra-domain loss
        intra_domain_loss = self._calc_intra_domain_loss(encoder_cls, premise, domain_ids)

        # Task loss
        image_encoder_output = self.image_encoder(pixel_values=pixel_values)["last_hidden_state"]
        text_encoder_output = self.text_encoder(input_ids=text_encoder_input["input_ids"],
                                                attention_mask=text_encoder_input["attention_mask"],
                                                token_type_ids=text_encoder_input["token_type_ids"])["last_hidden_state"][:, 0, :] # CLS token
        image_encoder_output = image_encoder_output[:, 0, :] # CLS token

        combined_output = torch.cat((image_encoder_output, text_encoder_output), dim=1)
        logits = self.classifier(combined_output)

        target_tensor = torch.tensor(label, dtype=torch.long, device=self.args.device)
        loss = self.loss_func(logits, target_tensor)

        final_loss = self.loss_func(logits, target_tensor) + self.args.inter_domain_loss_weight * inter_domain_loss + self.args.intra_domain_loss_weight * intra_domain_loss

        output = {
            'logits': logits,
            'loss': final_loss
        }
        return output

    def generate(self, image, premise, hypothesis):
        transformed_image = [self.transform_train(img) for img in image]

        pixel_values = self.image_processor(images=transformed_image, return_tensors="pt",
                                              do_resize=False, do_rescale=True, do_normalize=True).pixel_values
        text_encoder_input = self.text_tokenizer(hypothesis, padding="longest", return_tensors="pt")

        pixel_values = pixel_values.to(self.args.device)
        text_encoder_input = {key: val.to(self.args.device) for key, val in text_encoder_input.items()}

        image_encoder_output = self.image_encoder(pixel_values=pixel_values)["last_hidden_state"]
        text_encoder_output = self.text_encoder(input_ids=text_encoder_input["input_ids"],
                                                attention_mask=text_encoder_input["attention_mask"],
                                                token_type_ids=text_encoder_input["token_type_ids"])["last_hidden_state"][:, 0, :] # CLS token
        image_encoder_output = image_encoder_output[:, 0, :] # CLS token

        combined_output = torch.cat((image_encoder_output, text_encoder_output), dim=1)
        logits = self.classifier(combined_output)
        preds = logits.argmax(-1).tolist()

        return preds
