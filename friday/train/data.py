import os
import random
from dataclasses import dataclass
import json
from typing import Dict, Sequence

import torch

import transformers

from friday.model import SPECIAL_TOKENS
from friday.constants import IGNORE_INDEX, IMAGE_TOKEN, PAD_FOR_EOS
from torch.utils.data import Dataset

from PIL import Image


def preprocess_for_pretraining(
        sample: dict, 
        image_dir: str, 
        vision_tower: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizer
    ) -> dict:
    assert 'image' in sample and sample['image'] is not None, "sample provided without image"
    assert 'caption' in sample and sample['caption'] is not None, "sample provided without caption"

    img_files = sample.get("images") or [sample["image"]]
    assert len(img_files) > 0, "no image(s) provided for pre‑training"
    
    # 1) load and preprocess image
    preprocessed_images = []
    for img_file in img_files:
        image_path = os.path.join(image_dir, img_file)
        with Image.open(image_path) as im:
            image = vision_tower.preprocess_images(
                [im.convert("RGB")], pad_and_stack_tensors=False
            )[0]
        preprocessed_images.append(image)

    # 2) build the teacher‑forcing prompt: <image>  answer
    prompt = " ".join([IMAGE_TOKEN] * len(img_files)) + sample['caption']
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=tokenizer.model_max_length
    ).input_ids[0]

    # 3) clone for labels and mask the <image> token
    labels = input_ids.clone()
    labels = labels.masked_fill(labels == SPECIAL_TOKENS['image_token_id'], IGNORE_INDEX)

    return {
        "input_ids": input_ids, 
        "labels": labels,
        "image": preprocessed_images
    }



class PretrainingDataset(Dataset):
    """Dataset for aligning vision adapter."""

    def __init__(self, 
            data_path: str,
            image_dir: str,
            tokenizer: transformers.PreTrainedTokenizer,
            vision_tower,
            max_count: int = None
        ):
        super(PretrainingDataset, self).__init__()
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.samples = json.load(open(data_path, "r"))
        if max_count is not None:
            self.samples = random.sample(self.samples, max_count)
        
        self.sample_lengths = []
        for sample in self.samples:
            if 'caption' not in sample and 'blip_caption' in sample:
                sample['caption'] = sample.pop('blip_caption')
            
            img_tokens = self.vision_tower.num_patches if 'image' in sample else 0
            est_text_tokens = len(sample['caption'].split())
            total_tokens = img_tokens + est_text_tokens

            if total_tokens> self.tokenizer.model_max_length:
                total_tokens = self.tokenizer.model_max_length
            if img_tokens > 0:
                total_tokens *= -1 # negative length indicates a multimodal sample
            
            self.sample_lengths.append(total_tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return preprocess_for_pretraining(
            self.samples[i],
            self.image_dir,
            self.vision_tower,
            self.tokenizer
        )



@dataclass
class PretrainingCollator(object):
    """Collate examples for aligning vision adapter."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [ins["input_ids"].clone() for ins in instances]
        labels = [ins["labels"].clone()    for ins in instances]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            input_ids = [
                ids.masked_fill(ids == self.tokenizer.eos_token_id, PAD_FOR_EOS)
                for ids in input_ids
            ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX # ignore loss for padded positions
        )


        max_len = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_len]
        labels    =    labels[:, :max_len]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            input_ids.masked_fill_(input_ids == PAD_FOR_EOS, self.tokenizer.eos_token_id)

        images = [ins["image"] for ins in instances]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images
        )

