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
    
    def get_vision_tower(self) -> Optional[nn.Module]:
        return self.meta_model.get_vision_tower()
    
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
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        images: Optional[List[torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.LongTensor], Optional[torch.Tensor], Optional[List[torch.FloatTensor]], torch.Tensor, Optional[torch.Tensor]]:
        """Hybrid of Bunny (robust) + Friday (<img_start>/<img_end>)."""

        # ─────────────────── early return (no image / streaming step) ───────────────────
        vision_tower = self.model.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                tgt = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.size(0), tgt - attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)],
                    dim=1,
                )
                position_ids = attention_mask.sum(dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # ─────────────────────────── visual features (B, N, D) ───────────────────────────
        if isinstance(images, list) or images.ndim == 5:
            # concat = torch.cat(images, dim=0)
            image_features = self.encode_images(images).to(self.device)
            # splits = torch.split(feats, [img.shape[0] for img in images], dim=0)
            # image_features = [x.flatten(0, 1).to(self.device) for x in splits]
        else:
            image_features = self.encode_images(images).to(self.device)

        # ───────────────────────────── pad handling prelims ──────────────────────────────
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE)

        # strip current padding for efficiency
        input_ids_nopad = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels_nopad    = [lab[mask] for lab, mask in zip(labels, attention_mask)]

        # repetition‑penalty safety
        input_ids[input_ids == self.image_token_id] = 0

        # ───────────────────────────── splice per‑sample ────────────────────────────────
        embeds_list, labels_list = [], []
        img_ptr = 0
        for ids, labs in zip(input_ids_nopad, labels_nopad):
            positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            parts, lbl_parts = [], []
            cursor = 0
            for pos in positions:
                txt = ids[cursor:pos]
                parts.append(self.model.embed_tokens(txt))
                lbl_parts.append(txt)

                # start token
                parts.append(self.model.embed_tokens(ids.new_tensor([self.start_id])))
                lbl_parts.append(ids.new_tensor([IGNORE]))

                # visual tokens
                vis = image_features[img_ptr]
                img_ptr += 1
                parts.append(vis)
                lbl_parts.append(ids.new_tensor([IGNORE] * vis.size(0)))

                # end token
                parts.append(self.model.embed_tokens(ids.new_tensor([self.end_id])))
                lbl_parts.append(ids.new_tensor([IGNORE]))
                cursor = pos + 1
            # tail text
            tail = ids[cursor:]
            parts.append(self.model.embed_tokens(tail))
            lbl_parts.append(tail)

            embeds_list.append(torch.cat(parts))
            labels_list.append(torch.cat(lbl_parts))

        # ───────────────────── truncate then pad back to rectangle ──────────────────────
        max_ctx = getattr(self.config, 'tokenizer_model_max_length', None)
        if max_ctx is not None:
            embeds_list = [e[:max_ctx] for e in embeds_list]
            labels_list = [l[:max_ctx] for l in labels_list]

        max_len = max(e.size(0) for e in embeds_list)
        bs = len(embeds_list)
        emb_pad = torch.zeros(max_len, self.config.hidden_size, device=input_ids.device, dtype=self.dtype)
        lab_pad = torch.full((max_len,), IGNORE, device=input_ids.device, dtype=input_ids.dtype)

        new_input_embeds = torch.stack([torch.cat([e, emb_pad[e.size(0):]]) for e in embeds_list]).to(dtype=self.dtype)
        new_labels       = torch.stack([torch.cat([l, lab_pad[l.size(0):]]) for l in labels_list]) if labels is not None else None

        # rebuild mask & pos (right‑padding only)
        attention_mask = torch.arange(max_len, device=input_ids.device).expand(bs, -1) < torch.tensor([e.size(0) for e in embeds_list], device=input_ids.device).unsqueeze(1)
        position_ids   = torch.arange(max_len, device=input_ids.device).expand(bs, -1)

        if labels is None:
            new_labels = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

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
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
            # print(f"inputs_embeds.shape: {inputs_embeds.shape}")
            

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
    model.get_model().meta_model.load_model(device=model.device)
    # model.resize_token_embeddings(len(tok))

    return model, tok

AutoConfig.register("friday-phi", FridayPhiConfig)
AutoModelForCausalLM.register(FridayPhiConfig, FridayPhiForCausalLM)