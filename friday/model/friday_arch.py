from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import List, Tuple, Optional, Union

import PIL

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from friday.util import pad_and_stack, expand2square
from friday.model.vision_adapter import MLPAdapter
from friday.model.vision_tower import SiglipVisionTower, SiglipVisionTowerS2
from friday.model.language_model.phi4 import (
    Phi3Config, 
    Phi3Model, 
    Phi3ForCausalLM
)
from friday.constants import (
    IMAGE_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    IGNORE_INDEX
)

DEFAULT_CFG_SPECIAL_TOKENS = {
    "image_token_id": 200029,
    "image_start_token_id": 200030,
    "image_end_token_id": 200031,
}
DEFAULT_CFG_VISION_TOWER = {
    "vision_tower": "google/siglip2-base-patch16-384",
    # "vision_tower": "google/siglip2-so400m-patch16-384",
    "s2_scales": "384,768",
    "use_s2": True,
}
DEFAULT_CFG_VISION_ADAPTER = {
    "input_dim": 1536,
    "hidden_dim": 512,
    "output_dim": 3072,
    "layers": 2,
    "activation": "gelu"
}


class FridayConfig(Phi3Config):
    model_type = "friday-phi"

    def __init__(self, delay_load=True, **kwargs):
        super().__init__(**kwargs)

        self.delay_load = delay_load
        self.cfg_vision_tower = DEFAULT_CFG_VISION_TOWER.copy()
        if "cfg_vision_tower" in kwargs:
            self.cfg_vision_tower.update(kwargs["cfg_vision_tower"])

        self.cfg_vision_adapter = DEFAULT_CFG_VISION_ADAPTER.copy()
        if "cfg_vision_adapter" in kwargs:
            self.cfg_vision_adapter.update(kwargs["cfg_vision_adapter"])

        self.cfg_special_tokens = DEFAULT_CFG_SPECIAL_TOKENS.copy()
        if "cfg_special_tokens" in kwargs:
            self.cfg_special_tokens.update(kwargs["cfg_special_tokens"])


class FridayModel(Phi3Model):
    config_class = FridayConfig
    
    def __init__(self, config: FridayConfig):
        super().__init__(config)

        self.cfg_vision_adapter = config.cfg_vision_adapter
        self.cfg_vision_tower = config.cfg_vision_tower

        self.vision_tower = None
        self.mm_projector    = None
        if not config.delay_load:
            self.initialize_vision_modules()
    
    def get_vision_tower(self):
        return self.vision_tower
    
    def initialize_vision_modules(self):
        if self.cfg_vision_tower['use_s2']:
            self.vision_tower = SiglipVisionTowerS2(**self.cfg_vision_tower)
        else:
            self.vision_tower = SiglipVisionTower(**self.cfg_vision_tower)
        
        self.vision_tower.load_model(device_map=self.device)
        self.mm_projector = MLPAdapter(**self.cfg_vision_adapter).to(device=self.device)
    
    def compute_image_features(self, imgs: torch.Tensor) -> torch.Tensor:
        features = self.vision_tower(imgs)
        return self.mm_projector(features)

    def set_vision_adapter_requires_grad(self, requires_grad: bool):
        if self.mm_projector is not None:
            for param in self.mm_projector.parameters():
                param.requires_grad = requires_grad
        else:
            raise ValueError("Vision adapter is not initialized. Please call initialize_vision_modules() first.")
    
    def set_vision_tower_requires_grad(self, requires_grad: bool):
        if self.vision_tower is not None:
            for param in self.vision_tower.parameters():
                param.requires_grad = requires_grad
        else:
            raise ValueError("Vision tower is not initialized. Please call initialize_vision_modules() first.")


class FridayForCausalLM(Phi3ForCausalLM):
    config_class = FridayConfig

    def __init__(self, config: FridayConfig):
        super().__init__(config)
        self.config = config
        self.model = FridayModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.image_token_id = config.cfg_special_tokens["image_token_id"]
        self.start_id       = config.cfg_special_tokens["image_start_token_id"]
        self.end_id         = config.cfg_special_tokens["image_end_token_id"]
    
    def get_model(self) -> FridayModel:
        return self.model
    
    def get_vision_tower(self) -> SiglipVisionTower:
        return self.model.get_vision_tower()

    def set_language_model_requires_grad(self, requires_grad: bool):
        for param in self.model.parameters():
            param.requires_grad = requires_grad
        for param in self.lm_head.parameters():
            param.requires_grad = requires_grad
    
    def set_vision_tower_requires_grad(self, requires_grad: bool):
        self.model.set_vision_tower_requires_grad(requires_grad)

    def set_vision_adapter_requires_grad(self, requires_grad: bool):
        self.model.set_vision_adapter_requires_grad(requires_grad)
    
    def initialize_vision_modules(self):
        self.model.initialize_vision_modules()

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor,
        images: List[List[PIL.Image.Image]], # B x N
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.LongTensor], Optional[torch.Tensor], Optional[List[torch.FloatTensor]], torch.Tensor, Optional[torch.Tensor]]:
        
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

        # ─────────────────────────── images: (B, N) ───────────────────────────
        if isinstance(images, list) and isinstance(images[0], list):
            # images is a list of lists, each containing multiple images, B x N
            # e.g. [[img1, img2], [img3, img4]]
            assert len(images) == input_ids.shape[0], f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
            image_features = []
            for sublst_images in images:
                if len(sublst_images) == 0:
                    image_features.append(torch.zeros((0, self.get_model().mm_projector.output_dim), device=self.device))
                else:
                    if isinstance(sublst_images[0], PIL.Image.Image):
                        image_features.append(
                            self.model.compute_image_features(
                                self.model.vision_tower.preprocess_images(sublst_images)
                            )
                        )
                    elif isinstance(sublst_images[0], torch.Tensor):
                        # This should be a list of tensors of pre-processed images, [(N X 3 X W x H), ...]
                        image_features.append(
                            self.model.compute_image_features(sublst_images)
                        )
        elif isinstance(images, list) and isinstance(images[0], PIL.Image.Image):
            # images is a list of images for a single batch item, 1 x N
            # e.g. [img1, img2, img3]
            assert input_ids.shape[0] == 1, f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
            image_features = [
                self.model.compute_image_features(
                    self.model.vision_tower.preprocess_images(images)
                )
            ]
        elif isinstance(images, list) and isinstance(images[0], torch.Tensor):
            # This should be a list of tensors of pre-processed batch of images, [(N X 3 X W x H), ...]
            image_features = [
                self.model.compute_image_features(imgs) for imgs in images
            ]
        elif isinstance(images, PIL.Image.Image):
            # images is a single image, 1 x 1
            # e.g. img1
            assert input_ids.shape[0] == 1, f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
            image_features = [
                self.model.compute_image_features(
                    self.model.vision_tower.preprocess_images([images])
                )
            ]
        else:
            raise ValueError(f"Unsupported images format: {type(images)}. Expected list of PIL images, a single PIL image or a Tensor of pre-processed images")
        
        # ─────────────────────────── image_features: (B x N x D) ───────────────────────────
        if isinstance(image_features, list):
            assert input_ids.shape[0] == len(image_features), f"Incorrectly formatted image_features: list length should match batch size"
            assert isinstance(image_features[0], torch.Tensor), f"Incorrectly formatted image_features: list items should be tensors"
        elif isinstance(image_features, torch.Tensor):
            assert input_ids.shape[0] == image_features.shape[0], f"Incorrectly formatted image_features: tensor should match batch size"
        

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

            if len(positions) != image_features[batch_id].shape[0]:
                raise ValueError(
                    f"Mismatch between number of image tokens ({len(positions)}) and number of image features ({image_features[batch_id].shape[0]})"
                )

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

        is_multi_modal = images is not None and not (
            (
                isinstance(images, list) and (len(images) == 0 or all(i == [] for i in images))
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

    return model, tok


AutoConfig.register("friday-phi", FridayConfig)
AutoModelForCausalLM.register(FridayConfig, FridayForCausalLM)
