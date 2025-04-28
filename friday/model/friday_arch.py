from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import List, Tuple, Optional, Union

import PIL

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from friday.util import pad_and_stack
from friday.model.multi_modal_projector import MLPAdapter
from friday.model.vision_encoder import SiglipVisionTowerS2
from friday.model.language_model.phi4 import (
    Phi3Config, 
    Phi3Model, 
    Phi3ForCausalLM
)
from friday.util.constants import (
    IMAGE_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    IGNORE_INDEX
)


class FridayConfig(Phi3Config):
    model_type = "friday-phi"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg_vision_tower = {}
        self.cfg_vision_adapter = {}
        self.cfg_special_tokens = {}


class FridayModel(Phi3Model):
    config_class = FridayConfig
    
    def __init__(self, config: FridayConfig):
        super().__init__(config)

        self.cfg_vision_adapter = config.cfg_vision_adapter
        self.cfg_vision_tower = config.cfg_vision_tower
        self.vision_tower = None
        self.projector    = None
    
    def get_vision_tower(self):
        return self.vision_tower
    
    def initialize_vision_modules(self):
        self.vision_tower = SiglipVisionTowerS2(**self.cfg_vision_tower)
        self.vision_tower.load_model(device_map=self.device)
        self.projector = MLPAdapter(**self.cfg_vision_adapter).to(device=self.device)
    
    def encode_images(self, imgs: List[PIL.Image.Image]) -> torch.Tensor:
        img_tensors = [transforms.ToTensor()(img) for img in imgs]
        # print(f"self.vision_tower.device: {self.vision_tower.device}")
        imgs = pad_and_stack(img_tensors).to(dtype=torch.float32, device=self.vision_tower.device)
        features = self.vision_tower(imgs)

        return self.projector(features)
    
    def batch_encode_images(self, b_imgs: List[List[PIL.Image.Image]]) -> torch.Tensor:
        img_features = []
        for imgs in b_imgs:
            if len(imgs) == 0:
                img_features.append(torch.zeros((0, self.projector.output_dim), device=self.device))
            else:
                img_features.append(
                    self.encode_images(imgs)
                )
        
        return img_features

    def set_vision_projector_requires_grad(self, requires_grad: bool):
        for param in self.projector.parameters():
            param.requires_grad = requires_grad


class FridayForCausalLM(Phi3ForCausalLM):
    config_class = FridayConfig

    def __init__(self, config: FridayConfig):
        super().__init__(config)
        self.model = FridayModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.image_token_id = config.cfg_special_tokens["image"]
        self.start_id       = config.cfg_special_tokens["start"]
        self.end_id         = config.cfg_special_tokens["end"]
    
    def get_model(self) -> FridayModel:
        return self.model

    def encode_images(self, imgs: list) -> torch.Tensor:  # (B,3,H,W)
        return self.model.encode_images(imgs)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        images: List[List[PIL.Image.Image]], # B x N
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.LongTensor], Optional[torch.Tensor], Optional[List[torch.FloatTensor]], torch.Tensor, Optional[torch.Tensor]]:
        
        # print("**************")
        # print(images)
        # print("**************")

        # if images is None:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels


        # ─────────────────── early return (no image / streaming step) ───────────────────
        # if we have already processed images and are in a streaming step we can skip the multimodal processing
        # but we need to ensure the attention mask and position ids are correct
        if past_key_values is not None and input_ids.shape[1] == 1:
            tgt = past_key_values[-1][-1].shape[-2] + 1
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), tgt - attention_mask.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1,
            )
            position_ids = attention_mask.sum(dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # ─────────────────────────── visual features (B, N, D) ───────────────────────────
        if isinstance(images, list) and isinstance(images[0], list):
            # images is a list of lists, each containing multiple images, B x N
            # e.g. [[img1, img2], [img3, img4]]
            assert len(images) == input_ids.shape[0], f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
            image_features = self.model.batch_encode_images(images)
        elif isinstance(images, list) and isinstance(images[0], PIL.Image.Image):
            # images is a list of images, 1 x N
            # e.g. [img1, img2, img3]
            assert input_ids.shape[0] == 1, f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
            image_features = [self.encode_images(images).to(self.device)]
        elif isinstance(images, PIL.Image.Image):
            # images is a single image, 1 x 1
            # e.g. img1
            assert input_ids.shape[0] == 1, f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
            image_features = [self.encode_images([images]).to(self.device)]
        else:
            raise ValueError(f"Unsupported images format: {type(images)}. Expected list of PIL images or a single PIL image.")
        
        # print(f"image_features.shape: {image_features.shape}")

        # if isinstance(images, list) or images.ndim == 5:
        #     # concat = torch.cat(images, dim=0)
        #     image_features = self.encode_images(images).to(self.device)
        #     # splits = torch.split(feats, [img.shape[0] for img in images], dim=0)
        #     # image_features = [x.flatten(0, 1).to(self.device) for x in splits]
        # else:
        #     image_features = self.encode_images(images).to(self.device)

        # ───────────────────────────── pad handling prelims ──────────────────────────────
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # strip current padding for efficiency
        input_ids_nopad = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels_nopad    = [lab[mask] for lab, mask in zip(labels, attention_mask)]

        # repetition‑penalty safety
        input_ids[input_ids == self.image_token_id] = 0

        # ───────────────────────────── splice per‑sample ────────────────────────────────
        embeds_list, labels_list = [], []
        batch_id = 0
        for ids, labs in zip(input_ids_nopad, labels_nopad):
            positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            emb_parts, lbl_parts = [], []
            cursor, img_ptr = 0, 0
            for pos in positions:
                txt = ids[cursor:pos]
                emb_parts.append(self.model.embed_tokens(txt))
                lbl_parts.append(txt)

                # start token
                emb_parts.append(self.model.embed_tokens(ids.new_tensor([self.start_id])))
                lbl_parts.append(ids.new_tensor([IGNORE_INDEX]))

                # visual tokens
                vis = image_features[batch_id][img_ptr]
                emb_parts.append(vis)
                lbl_parts.append(ids.new_tensor([IGNORE_INDEX] * vis.shape[0]))

                # end token
                emb_parts.append(self.model.embed_tokens(ids.new_tensor([self.end_id])))
                lbl_parts.append(ids.new_tensor([IGNORE_INDEX]))

                img_ptr += 1
                cursor = pos + 1
            
            # tail text
            tail = ids[cursor:]
            emb_parts.append(self.model.embed_tokens(tail))
            lbl_parts.append(tail)

            embeds_list.append(torch.cat(emb_parts))
            labels_list.append(torch.cat(lbl_parts))
            batch_id += 1

        # ───────────────────── truncate then pad back to rectangle ──────────────────────
        max_ctx = getattr(self.config, 'tokenizer_model_max_length', None)
        if max_ctx is not None:
            embeds_list = [e[:max_ctx] for e in embeds_list]
            labels_list = [l[:max_ctx] for l in labels_list]

        max_len = max(e.size(0) for e in embeds_list)
        bs = len(embeds_list)
        emb_pad = torch.zeros(max_len, self.config.hidden_size, device=input_ids.device, dtype=self.dtype)
        lab_pad = torch.full((max_len,), IGNORE_INDEX, device=input_ids.device, dtype=input_ids.dtype)

        new_input_embeds = torch.stack([torch.cat([e, emb_pad[e.size(0):]]) for e in embeds_list]).to(dtype=self.dtype)
        new_labels       = torch.stack([torch.cat([l, lab_pad[l.size(0):]]) for l in labels_list]) if labels is not None else None

        # rebuild mask & pos (right‑padding only)
        attention_mask = torch.arange(max_len, device=input_ids.device).expand(bs, -1) < torch.tensor([e.size(0) for e in embeds_list], device=input_ids.device).unsqueeze(1)
        position_ids   = torch.arange(max_len, device=input_ids.device).expand(bs, -1)

        if not self.training:
            new_labels = None
        
        # print(f"new_input_embeds.shape: {new_input_embeds.shape}")

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
            images: Optional[PIL.Image.Image] = None,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print(f"input_ids.shape: {input_ids.shape if input_ids is not None else 'None'}")
        # print(input_ids)

        is_multi_modal = images is not None and (
            (
                isinstance(images, list) and len(images) > 0 and any(i for i in images)
            ) or
            (
                isinstance(images, PIL.Image.Image)
            )
        )

        if inputs_embeds is None and is_multi_modal:
            # print("Preparing multimodal inputs...")
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                images=images,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
            )

            # print(f"inputs_embeds.shape: {inputs_embeds.shape}")
            if cache_position is not None and inputs_embeds is not None and cache_position.shape[0] != inputs_embeds.shape[1]:
                cache_position = torch.arange(inputs_embeds.shape[1], device=self.device)
            

        return Phi3ForCausalLM.forward(
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
    specials = {t: tok.convert_tokens_to_ids(t) for t in [IMAGE_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN] if t in tok.vocab}
    if len(specials) < 3:
        n = tok.add_tokens([IMAGE_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN], special_tokens=True)
        tok.pad_token = tok.eos_token
        specials = {
            "image": tok.convert_tokens_to_ids(IMAGE_TOKEN),
            "start": tok.convert_tokens_to_ids(IMG_START_TOKEN),
            "end": tok.convert_tokens_to_ids(IMG_END_TOKEN),
        }
    return tok, specials


def build_friday_phi(config: dict, base_lm: str = "microsoft/Phi-4-mini-instruct") -> FridayForCausalLM:
    tok, special_tokens = build_tokenizer(base_lm)

    base_cfg = FridayConfig.from_pretrained(base_lm)
    base_cfg.cfg_vision_tower = config.get("vision_tower", {})
    base_cfg.cfg_vision_adapter = config.get("vision_adapter", {})
    base_cfg.cfg_special_tokens = special_tokens

    model = FridayForCausalLM.from_pretrained(
        base_lm, 
        low_cpu_mem_usage=True,
        config=base_cfg, 
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.get_model().initialize_vision_modules()
    # model.resize_token_embeddings(len(tok))

    # print(f"model.device: {model.device}")
    # print(f"model.get_model().device: {model.get_model().device}")
    # print(f"model.vision_tower.device: {model.get_model().vision_tower.device}")
    # print(f"model.projector.device: {model.get_model().projector.device}")

    return model, tok


AutoConfig.register("friday-phi", FridayConfig)
AutoModelForCausalLM.register(FridayConfig, FridayForCausalLM)
