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

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', '']

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

    def _update_memory_bank(self, memory_cls, memory_domain_id, caption, caption_embedding, caption_pos_tag):
        if self.memory_bank['memory_cls'] is None:
            self.memory_bank['memory_cls'] = memory_cls
            self.memory_bank['memory_domain_id'] = memory_domain_id
            self.memory_bank['caption'] = caption
            self.memory_bank['caption_embedding'] = caption_embedding
            self.memory_bank['caption_pos_tag'] = caption_pos_tag
        else:
            self.memory_bank['memory_cls'] = torch.cat([self.memory_bank['memory_cls'], memory_cls], dim=0)
            self.memory_bank['memory_domain_id'].extend(memory_domain_id)
            self.memory_bank['caption'].extend(caption)
            self.memory_bank['caption_embedding'].extend(caption_embedding)
            self.memory_bank['caption_pos_tag'].extend(caption_pos_tag)

        if len(self.memory_bank['memory_cls']) > self.args.memory_bank_size:
            self.memory_bank['memory_cls'] = self.memory_bank['memory_cls'][-self.args.memory_bank_size:]
            self.memory_bank['memory_domain_id'] = self.memory_bank['memory_domain_id'][-self.args.memory_bank_size:]
            self.memory_bank['caption'] = self.memory_bank['caption'][-self.args.memory_bank_size:]
            self.memory_bank['caption_embedding'] = self.memory_bank['caption_embedding'][-self.args.memory_bank_size:]
            self.memory_bank['caption_pos_tag'] = self.memory_bank['caption_pos_tag'][-self.args.memory_bank_size:]

    def _update_momentum_encoder(self):
        for param, momentum_param in zip(self.image_encoder.parameters(), self.momentum_encoder.parameters()):
            momentum_param.data = momentum_param.data * self.args.momentum_coefficient + param.data * (1 - self.args.momentum_coefficient)

    def _calc_sentence_similarity(self, anchor_caption, target_caption, target_embedding=None):
        if self.anchor_caption == None or self.anchor_embedding == None:
            self.anchor_caption = anchor_caption
            self.anchor_embedding = self.sentence_bert.encode(anchor_caption, convert_to_tensor=True)
        elif self.anchor_caption != anchor_caption:
            self.anchor_caption = anchor_caption
            self.anchor_embedding = self.sentence_bert.encode(anchor_caption, convert_to_tensor=True)

        if target_embedding == None:
            target_embedding = self.sentence_bert.encode(target_caption, convert_to_tensor=True)
        sentence_similarity = round(cos_sim(self.anchor_embedding, target_embedding).item(), 4) * self.args.sentence_sim_scaling

        return sentence_similarity, target_embedding

    def _calc_visual_word_similarity(self, anchor_caption=None, target_caption=None, target_pos_tag=None):
        if self.anchor_caption == None or self.anchor_pos_tag == None:
            self.anchor_caption = anchor_caption
            self.anchor_pos_tag = self.pos_tagger(anchor_caption)
            self.anchor_pos_tag = [token.text for token in self.anchor_pos_tag if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            self.anchor_pos_tag = [word for word in self.anchor_pos_tag if word not in stop_words]
        elif self.anchor_caption != anchor_caption:
            self.anchor_caption = anchor_caption
            self.anchor_pos_tag = self.pos_tagger(anchor_caption)
            self.anchor_pos_tag = [token.text for token in self.anchor_pos_tag if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            self.anchor_pos_tag = [word for word in self.anchor_pos_tag if word not in stop_words]

        if target_pos_tag == None:
            target_pos_tag = self.pos_tagger(target_caption)
            target_pos_tag = [token.text for token in target_pos_tag if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            target_pos_tag = [word for word in target_pos_tag if word not in stop_words]

        try:
            visual_word_IoU = round(len(set(self.anchor_pos_tag) & set(target_pos_tag)) /
                                    len(set(self.anchor_pos_tag) | set(target_pos_tag)), 4)
        except ZeroDivisionError:
            visual_word_IoU = 0

        return visual_word_IoU, target_pos_tag

    def _get_momentum_encoding(self, pixel_values):
        with torch.no_grad():
            momentum_encoder_outputs = self.momentum_encoder(pixel_values=pixel_values)
        encoder_hidden_states = momentum_encoder_outputs.last_hidden_state
        encoder_cls = encoder_hidden_states[:, 0, :]
        return encoder_cls

    def _get_pos_neg_idx(self, visual_word_sim_list, sentence_sim_list):
        assert len(visual_word_sim_list) == len(sentence_sim_list)

        final_sim = [max(visual_word_sim_list[i], sentence_sim_list[i])
                     for i in range(len(visual_word_sim_list))]

        pos_idx = [i for i in range(len(final_sim))
                   if final_sim[i] >= self.args.positive_threshold]
        neg_idx = [i for i in range(len(final_sim))
                   if self.args.negative_lower_bound <= final_sim[i] < self.args.negative_upper_bound]

        return pos_idx, neg_idx

    def _calc_inter_domain_loss(self, anchor_cls, anchor_caption):
        inter_domain_loss = 0
        # use values from memory bank to get positive/negative samples for inter-domain loss
        inter_visual_word_similarity = [self._calc_visual_word_similarity(anchor_caption, self.memory_bank['caption'][i], self.memory_bank['caption_pos_tag'][i])
                                        for i in range(len(self.memory_bank['caption']))]
        inter_sentence_similarity = [self._calc_sentence_similarity(anchor_caption, self.memory_bank['caption'][i], self.memory_bank['caption_embedding'][i])
                                     for i in range(len(self.memory_bank['caption']))]
        inter_visual_word_similarity = [similarity[0] for similarity in inter_visual_word_similarity] # Discard the target_pos_tag
        inter_sentence_similarity = [similarity[0] for similarity in inter_sentence_similarity] # Discard the target_embedding

        inter_pos_idx, inter_neg_idx = self._get_pos_neg_idx(inter_visual_word_similarity, inter_sentence_similarity)
        if len(inter_pos_idx) == 0 or len(inter_neg_idx) == 0:
            return 0

        for i in range(len(inter_pos_idx)):
            for j in range(len(inter_neg_idx)):
                positive_dist = torch.square(anchor_cls - self.memory_bank['memory_cls'][inter_pos_idx[i]]).sum()
                negative_dist = torch.square(anchor_cls - self.memory_bank['memory_cls'][inter_neg_idx[j]]).sum()
                inter_domain_loss += torch.clamp(negative_dist - positive_dist + self.args.triplet_margin, min=0)
        inter_domain_loss /= len(inter_pos_idx) * len(inter_neg_idx)

        return inter_domain_loss

    def _calc_intra_domain_loss(self, encoder_cls, caption, domain_ids):
        intra_domain_loss = 0
        domain_ids_list = list(set(domain_ids)) # Get unique domain ids

        for each_domain in domain_ids_list:
            each_domain_loss = 0

            domain_cls = encoder_cls[[domain_ids[i] == each_domain for i in range(len(caption))], :]
            domain_caption = [caption[i] for i in range(len(caption)) if domain_ids[i] == each_domain]
            domain_anchor_cls = domain_cls[0, :]
            domain_memory_bank_caption = [self.memory_bank['caption'][i] for i in range(len(self.memory_bank['caption']))
                                          if self.memory_bank['memory_domain_id'][i] == each_domain]
            domain_memory_bank_cls = self.memory_bank['memory_cls'][[self.memory_bank['memory_domain_id'][i] == each_domain for i in range(len(self.memory_bank['memory_domain_id']))], :]
            domain_memory_bank_caption_embedding = [self.memory_bank['caption_embedding'][i] for i in range(len(self.memory_bank['caption_embedding']))
                                                    if self.memory_bank['memory_domain_id'][i] == each_domain]
            domain_memory_bank_caption_pos_tag = [self.memory_bank['caption_pos_tag'][i] for i in range(len(self.memory_bank['caption_pos_tag']))
                                                  if self.memory_bank['memory_domain_id'][i] == each_domain]
            assert len(domain_memory_bank_caption) == len(domain_memory_bank_cls) == len(domain_memory_bank_caption_embedding) == len(domain_memory_bank_caption_pos_tag)

            domain_visual_word_similarity = [self._calc_visual_word_similarity(domain_caption[0], domain_memory_bank_caption[i], domain_memory_bank_caption_pos_tag[i])
                                             for i in range(len(domain_memory_bank_caption))]
            domain_sentence_similarity = [self._calc_sentence_similarity(domain_caption[0], domain_memory_bank_caption[i], domain_memory_bank_caption_embedding[i])
                                          for i in range(len(domain_memory_bank_caption))]
            domain_visual_word_similarity = [similarity[0] for similarity in domain_visual_word_similarity] # Discard the target_pos_tag
            domain_sentence_similarity = [similarity[0] for similarity in domain_sentence_similarity] # Discard the target_embedding

            domain_pos_idx, domain_neg_idx = self._get_pos_neg_idx(domain_visual_word_similarity, domain_sentence_similarity)
            if len(domain_pos_idx) == 0 or len(domain_neg_idx) == 0:
                continue

            for i in range(len(domain_pos_idx)):
                for j in range(len(domain_neg_idx)):
                    positive_dist = torch.square(domain_anchor_cls - domain_memory_bank_cls[domain_pos_idx[i]]).sum()
                    negative_dist = torch.square(domain_anchor_cls - domain_memory_bank_cls[domain_neg_idx[j]]).sum()
                    each_domain_loss += torch.clamp(negative_dist - positive_dist + self.args.triplet_margin, min=0)
            each_domain_loss /= len(domain_pos_idx) * len(domain_neg_idx)
            intra_domain_loss += each_domain_loss

        intra_domain_loss /= len(domain_ids_list)
        return intra_domain_loss
