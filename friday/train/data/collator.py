from dataclasses import dataclass
from typing import Dict, Sequence

import torch

import transformers

from friday.util import mask_token_segment
from friday.constants import IGNORE_INDEX, PAD_FOR_EOS


@dataclass
class FridayCollator(object):
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

