import os
import random
from dataclasses import dataclass
import json
from typing import Dict, Sequence

import torch

import transformers

from friday.util import mask_token_segment
from friday.model import SPECIAL_TOKENS
from friday.constants import IGNORE_INDEX, IMAGE_TOKEN, PAD_FOR_EOS
from torch.utils.data import Dataset

from PIL import Image


def preprocess_for_finetuning(
        sample: dict, 
        image_dir: str, 
        vision_tower: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizer,
        system_message: str = None
    ) -> dict:
    assert 'conversations' in sample and sample['conversations'], "sample provided without conversation"

    # 1) load and preprocess images
    img_files = sample.get("images") or ([sample["image"]] if "image" in sample and sample['image'] is not None else [])
    preprocessed_images = []
    for img_file in img_files:
        image_path = os.path.join(image_dir, img_file)
        with Image.open(image_path) as im:
            image = vision_tower.preprocess_images(
                [im.convert("RGB")], pad_and_stack_tensors=False
            )[0]
        preprocessed_images.append(image)
    

    # 2) build the teacher‑forcing prompt
    prompt = f"<|system|>{system_message}<|end|>" if system_message else ""
    for conv in sample['conversations']:
        if conv['from'] == 'human':
            prompt += f"<|user|>{conv['value']}<|end|>"
        elif conv['from'] == 'gpt':
            prompt += f"<|assistant|>{conv['value']}<|end|>"
        else:
            raise ValueError(f"Unknown conversation type: {conv['from']}")

    image_token_count = prompt.count(IMAGE_TOKEN)
    if image_token_count != len(preprocessed_images):
        raise ValueError(
            f"Image token count ({image_token_count}) does not match number of images ({len(preprocessed_images)})"
        )
    
    max_length = getattr(tokenizer, "model_max_length", None)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_length
    ).input_ids[0]


    # 3) mask non-assistant token labels and special tokens
    system_token = tokenizer("<|system|>")["input_ids"][0]
    user_token = tokenizer("<|user|>")["input_ids"][0]
    assistant_token = tokenizer("<|assistant|>")["input_ids"][0]
    end_token = tokenizer("<|end|>")["input_ids"][0]

    labels = input_ids.clone()
    
    labels = mask_token_segment(system_token, end_token, labels, IGNORE_INDEX)
    labels = mask_token_segment(user_token, end_token, labels, IGNORE_INDEX)
    labels = labels.masked_fill(
        torch.isin(labels, torch.tensor([system_token, user_token, end_token, assistant_token])),
        IGNORE_INDEX
    )
    labels[-1] = end_token


    return {
        "input_ids": input_ids, 
        "labels": labels,
        "image": preprocessed_images
    }



class FinetuningDataset(Dataset):
    """Dataset for aligning vision adapter."""

    def __init__(self, 
            data_path: str,
            image_dir: str,
            tokenizer: transformers.PreTrainedTokenizer,
            vision_tower,
            max_count: int = None
        ):
        super(FinetuningDataset, self).__init__()
        
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
            est_text_tokens = sum([len(c['value']) for c in sample['conversations']])
            total_tokens = img_tokens + est_text_tokens

            if total_tokens> self.tokenizer.model_max_length:
                total_tokens = self.tokenizer.model_max_length
            if img_tokens > 0:
                total_tokens *= -1 # negative length indicates a multimodal sample
            
            self.sample_lengths.append(total_tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return preprocess_for_finetuning(
            self.samples[i],
            self.image_dir,
            self.vision_tower,
            self.tokenizer
        )

