from __future__ import annotations

from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .phi4 import (
    Phi3Config as PhiConfig, 
    Phi3Model as PhiModel, 
    Phi3ForCausalLM as PhiForCausalLM
)

from ..friday_arch import FridayMetaModel

IMAGE_TOKEN = "<image>"
IMG_START   = "<img_start>"
IMG_END     = "<img_end>"
IGNORE      = -100



class FridayPhiConfig(PhiConfig):
    model_type = "friday-phi"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg_vision_tower = {}
        self.cfg_vision_adapter = {}
        self.cfg_special_tokens = {}

class FridayPhiModel(PhiModel):
    config_class = FridayPhiConfig
    
    def __init__(self, config: FridayPhiConfig):
        super().__init__(config)

        self.meta_model = FridayMetaModel(
            cfg_vision_tower=config.cfg_vision_tower,
            cfg_vision_adapter=config.cfg_vision_adapter
        )
    
    def encode_images(self, imgs: List[torch.Tensor]) -> torch.Tensor:
        return self.meta_model.encode_images(imgs)

class FridayPhiForCausalLM(PhiForCausalLM):
    config_class = FridayPhiConfig

    def __init__(self, config: FridayPhiConfig):
        super().__init__(config)
        self.model = FridayPhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.image_token_id = config.cfg_special_tokens["image"]
        self.start_id       = config.cfg_special_tokens["start"]
        self.end_id         = config.cfg_special_tokens["end"]
    
    def get_model(self) -> FridayPhiModel:
        return self.model

    def encode_images(self, imgs: list) -> torch.Tensor:  # (B,3,H,W)
        return self.model.encode_images(imgs)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        images: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        if images is None or (input_ids == self.image_token_id).sum() == 0:
            return self.embed_tokens(input_ids), labels  # type: ignore

        B = input_ids.size(0)
        assert len(images) == B, "#images must equal batch"
        img_embeds = self.model.encode_images(images)  # (B,N,3072)

        all_embeds, all_labels = [], []
        for i in range(B):
            ids = input_ids[i]
            img_tokens = img_embeds[i]
            parts, lbls = [], []
            ptr = 0
            for pos in (ids == self.image_token_id).nonzero(as_tuple=True)[0]:
                # text before
                txt_ids = ids[ptr:pos]
                parts.append(self.embed_tokens(txt_ids))  # type: ignore
                lbls.append(txt_ids)
                # image block
                parts.append(self.embed_tokens(ids.new_tensor([self.start_id])))
                parts.append(img_tokens)
                parts.append(self.embed_tokens(ids.new_tensor([self.end_id])))
                lbls.append(ids.new_tensor([-100] * (1 + img_tokens.size(0) + 1)))
                ptr = pos + 1
            # tail text
            txt_ids = ids[ptr:]
            parts.append(self.embed_tokens(txt_ids))
            lbls.append(txt_ids)
            all_embeds.append(torch.cat(parts))
            all_labels.append(torch.cat(lbls))

        # pad to same length
        L = max(x.size(0) for x in all_embeds)
        pad_emb = torch.zeros(L, self.config.hidden_size, device=input_ids.device, dtype=self.dtype)
        pad_lab = torch.full((L,), IGNORE, device=input_ids.device, dtype=input_ids.dtype)
        embeds = torch.stack([torch.cat([e, pad_emb[e.size(0):]]) for e in all_embeds])
        labels = torch.stack([torch.cat([l, pad_lab[l.size(0):]]) for l in all_labels]) if labels is not None else None
        return embeds, labels

    # ------------------------------------------------------------------
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            images: Optional[torch.FloatTensor] = None,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None and images is not None:
            print("Preparing multimodal inputs...")
            (
                # input_ids,
                # position_ids,
                # attention_mask,
                # past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                # position_ids,
                # attention_mask,
                # past_key_values,
                labels=labels,
                images=images
            )

        return PhiForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )



def build_tokenizer(base_model_id: str) -> Tuple[AutoTokenizer, dict]:
    tok = AutoTokenizer.from_pretrained(base_model_id, padding_side="right")
    specials = {t: tok.convert_tokens_to_ids(t) for t in [IMAGE_TOKEN, IMG_START, IMG_END] if t in tok.vocab}
    if len(specials) < 3:
        n = tok.add_tokens([IMAGE_TOKEN, IMG_START, IMG_END], special_tokens=True)
        tok.pad_token = tok.eos_token
        specials = {
            "image": tok.convert_tokens_to_ids(IMAGE_TOKEN),
            "start": tok.convert_tokens_to_ids(IMG_START),
            "end": tok.convert_tokens_to_ids(IMG_END),
        }
    return tok, specials


def build_friday_phi(config: dict, base_lm: str = "microsoft/Phi-4-mini-instruct") -> FridayPhiForCausalLM:
    tok, special_tokens = build_tokenizer(base_lm)

    base_cfg = FridayPhiConfig.from_pretrained(base_lm)
    base_cfg.cfg_vision_tower = config.get("vision_tower", {})
    base_cfg.cfg_vision_adapter = config.get("vision_adapter", {})
    base_cfg.cfg_special_tokens = special_tokens

    model = FridayPhiForCausalLM.from_pretrained(
        base_lm, 
        low_cpu_mem_usage=True,
        config=base_cfg, 
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
    # model.resize_token_embeddings(len(tok))

    return model, tok

AutoConfig.register("friday-phi", FridayPhiConfig)
AutoModelForCausalLM.register(FridayPhiConfig, FridayPhiForCausalLM)