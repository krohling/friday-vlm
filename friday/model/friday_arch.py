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
    "pretrained_model_name_or_path": "google/siglip2-base-patch16-384",
    # "pretrained_model_name_or_path": "google/siglip2-so400m-patch16-384",
    "s2_scales": "384,768",
    "use_s2": True,
    "pad_to_square": True,
    "freeze": False,
}
DEFAULT_CFG_VISION_ADAPTER = {
    "input_dim": 1536,
    "hidden_dim": 512,
    "output_dim": 3072,
    "layers": 2,
    "activation": "gelu",
    "freeze": False,
}


class FridayConfig(Phi3Config):
    model_type = "friday-phi"

    def __init__(self, 
            base_model_name_or_path: str | None = "microsoft/Phi-4-mini-instruct",
            delay_load=True, 
            tokenizer_model_max_length=None,
            **kwargs
        ):
        base_kwargs = {}
        if base_model_name_or_path is not None:
            base_cfg = AutoConfig.from_pretrained(
                base_model_name_or_path,
                trust_remote_code=True,   # Phi‑4 uses custom code in the repo
            )
            base_kwargs = base_cfg.to_dict()

        merged = {**base_kwargs, **kwargs}
        self.delay_load = delay_load
        self.tokenizer_model_max_length = tokenizer_model_max_length

        self._cfg_vision_tower = DEFAULT_CFG_VISION_TOWER.copy()
        if "cfg_vision_tower" in kwargs:
            self._cfg_vision_tower.update(kwargs["cfg_vision_tower"])

        self._cfg_vision_adapter = DEFAULT_CFG_VISION_ADAPTER.copy()
        if "cfg_vision_adapter" in kwargs:
            self._cfg_vision_adapter.update(kwargs["cfg_vision_adapter"])

        self._cfg_special_tokens = DEFAULT_CFG_SPECIAL_TOKENS.copy()
        if "cfg_special_tokens" in kwargs:
            self._cfg_special_tokens.update(kwargs["cfg_special_tokens"])

        super().__init__(**merged)
        
    
    @property
    def cfg_vision_tower(self):
        return self._cfg_vision_tower

    @cfg_vision_tower.setter
    def cfg_vision_tower(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._cfg_vision_tower.update(value)
    

    @property
    def cfg_vision_adapter(self):
        return self._cfg_vision_adapter
    
    @cfg_vision_adapter.setter
    def cfg_vision_adapter(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._cfg_vision_adapter.update(value)
    
    @property
    def cfg_special_tokens(self):
        return self._cfg_special_tokens
    
    @cfg_special_tokens.setter
    def cfg_special_tokens(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._cfg_special_tokens.update(value)


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
        if self.vision_tower is not None:
            return

        if self.cfg_vision_tower.get("use_s2", True):
            self.vision_tower = SiglipVisionTowerS2(**self.cfg_vision_tower)
        else:
            self.vision_tower = SiglipVisionTower(**self.cfg_vision_tower)
        
        self.vision_tower.load_model()
        self.mm_projector = MLPAdapter(**self.cfg_vision_adapter)

        self.set_vision_tower_requires_grad(not self.cfg_vision_tower["freeze"])
        self.set_vision_adapter_requires_grad(not self.cfg_vision_adapter["freeze"])
    
    def compute_image_features(self, imgs: torch.Tensor) -> torch.Tensor:
        features = self.vision_tower(imgs)
        if isinstance(features, list):
            features = torch.stack(features, dim=1)
        return self.mm_projector(features)
    
    def set_vision_tower_requires_grad(self, requires_grad: bool):
        if self.vision_tower is not None:
            for param in self.vision_tower.parameters():
                param.requires_grad = requires_grad
        else:
            raise ValueError("Vision tower is not initialized. Please call initialize_vision_modules() first.")
    
    def set_vision_adapter_requires_grad(self, requires_grad: bool):
        if self.mm_projector is not None:
            for param in self.mm_projector.parameters():
                param.requires_grad = requires_grad
        else:
            raise ValueError("Vision adapter is not initialized. Please call initialize_vision_modules() first.")
    
    def set_vision_tower_dtype(self, dtype: torch.dtype):
        if self.vision_tower is not None:
            for p in self.vision_tower.parameters():
                p.data = p.data.to(dtype)
        else:
            raise ValueError("Vision tower is not initialized. Please call initialize_vision_modules() first.")
    
    def set_vision_adapter_dtype(self, dtype: torch.dtype):
        if self.mm_projector is not None:
            for p in self.mm_projector.parameters():
                p.data = p.data.to(dtype)
        else:
            raise ValueError("Vision adapter is not initialized. Please call initialize_vision_modules() first.")
    
    def is_vision_tower_frozen(self):
        if self.vision_tower is not None:
            return all(not p.requires_grad for p in self.vision_tower.parameters())
        else:
            raise ValueError("Vision tower is not initialized. Please call initialize_vision_modules() first.")
    
    def is_vision_adapter_frozen(self):
        if self.mm_projector is not None:
            return all(not p.requires_grad for p in self.mm_projector.parameters())
        else:
            raise ValueError("Vision adapter is not initialized. Please call initialize_vision_modules() first.")


class FridayForCausalLM(Phi3ForCausalLM):
    config_class = FridayConfig

    def __init__(self, config: FridayConfig):
        super().__init__(config)

        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_token_id = config.cfg_special_tokens["image_token_id"]
        self.image_start_id       = config.cfg_special_tokens["image_start_token_id"]
        self.image_end_id         = config.cfg_special_tokens["image_end_token_id"]

        self.model = FridayModel(config)
        self.post_init()
    
    def get_model(self) -> FridayModel:
        return self.model
    
    def get_vision_tower(self) -> SiglipVisionTower:
        return self.model.get_vision_tower()
    
    def get_vision_adapter(self) -> MLPAdapter:
        return self.model.mm_projector

    def get_llm_parameters(self):
        return [p for n, p in self.named_parameters() if "vision_tower" not in n and "mm_projector" not in n]

    def set_language_model_requires_grad(self, requires_grad: bool):
        for p in self.get_llm_parameters():
            p.requires_grad = requires_grad
    
    def set_vision_tower_requires_grad(self, requires_grad: bool):
        self.model.set_vision_tower_requires_grad(requires_grad)

    def set_vision_adapter_requires_grad(self, requires_grad: bool):
        self.model.set_vision_adapter_requires_grad(requires_grad)
    
    def set_language_model_dtype(self, dtype: torch.dtype):
        for p in self.get_llm_parameters():
            p.data = p.data.to(dtype)

    def set_vision_tower_dtype(self, dtype: torch.dtype):
        self.model.set_vision_tower_dtype(dtype)
    
    def set_vision_adapter_dtype(self, dtype: torch.dtype):
        self.model.set_vision_adapter_dtype(dtype)
    
    def is_llm_frozen(self):
        return all(not p.requires_grad for p in self.get_llm_parameters())
    
    def is_vision_tower_frozen(self):
        return self.model.is_vision_tower_frozen()
    
    def is_vision_adapter_frozen(self):
        return self.model.is_vision_adapter_frozen()
    
    
    
    def initialize_vision_modules(self):
        self.model.initialize_vision_modules()
    
    def get_multimodal_input_embeddings(self, input_ids, image_features, return_labels=True) -> torch.Tensor:
        emb_start_image_id = self.model.embed_tokens(torch.tensor([self.image_start_id], device=self.device))
        emb_end_image_id   = self.model.embed_tokens(torch.tensor([self.image_end_id], device=self.device))
        id_ignore = torch.tensor([IGNORE_INDEX], device=self.device)

        # repetition‑penalty safety ????
        # input_ids[input_ids == self.image_token_id] = 0

        
        # Iterate over each batch item
        embeds_list, labels_list = [], []
        for batch_id, item_ids in enumerate(input_ids):
            
            image_token_positions = (item_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(image_token_positions) != image_features[batch_id].shape[0]:
                raise ValueError(
                    f"Mismatch between number of image tokens ({len(image_token_positions)}) and number of image features ({image_features[batch_id].shape[0]})"
                )


            cursor = 0
            emb_parts, lbl_parts = [], []
            for indx_image, image_token_pos in enumerate(image_token_positions):
                if image_token_pos > cursor:
                    span = item_ids[cursor:image_token_pos]
                    emb_parts.append(self.model.embed_tokens(span))
                    lbl_parts.append(span)

                # <image_start>
                emb_parts.append(emb_start_image_id)
                lbl_parts.append(id_ignore)

                # vision embeddings
                image_tokens = image_features[batch_id][indx_image]
                if image_tokens.shape[0] == 1 and image_tokens.ndim == 3:
                    image_tokens = image_tokens.squeeze(0)
                emb_parts.append(image_tokens)
                lbl_parts.append(id_ignore.repeat(image_tokens.shape[0]))

                # <image_end>
                emb_parts.append(emb_end_image_id)
                lbl_parts.append(id_ignore)

                cursor = image_token_pos + 1
            
            # tail text
            if cursor < item_ids.shape[0]:
                tail = item_ids[cursor:]
                emb_parts.append(self.model.embed_tokens(tail))
                lbl_parts.append(tail)
            
            embeds_list.append(torch.cat(emb_parts, dim=0))
            labels_list.append(torch.cat(lbl_parts, dim=0))
    
        return (embeds_list, labels_list) if return_labels else embeds_list

    def prepare_inputs_for_multimodal(
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

        if past_key_values is not None and attention_mask is not None and input_ids.shape[1] == 1:
            tgt = past_key_values[-1][-1].shape[-2] + 1
            attention_mask = torch.cat(
                [attention_mask,
                torch.ones((attention_mask.size(0),
                            tgt - attention_mask.size(1)),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device)],
                dim=1,
            )
            position_ids = (attention_mask.sum(dim=1, keepdim=True) - 1).long()

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
                                self.model.vision_tower.preprocess_images(sublst_images, pad_and_stack_tensors=True)
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
                    self.model.vision_tower.preprocess_images(images, pad_and_stack_tensors=True)
                )
            ]
        elif isinstance(images, list) and isinstance(images[0], torch.Tensor):
            # This should be a list of tensors of pre-processed images, [(N X 3 X W x H), ...]
            # The list length should match the batch size
            assert input_ids.shape[0] == len(images), f"Batch size mismatch: {len(images)} vs {input_ids.shape[0]}"
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

        input_ids_nopad = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        embeds_list, labels_list = self.get_multimodal_input_embeddings(
            input_ids_nopad,
            image_features,
            return_labels=True
        )

        # ───────────────────── truncate then pad back to rectangle ──────────────────────
        new_input_embeds = torch.nn.utils.rnn.pad_sequence(
            embeds_list,
            batch_first=True,
            padding_value=0.0
        ).to(dtype=self.dtype)

        new_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX
        ).long()

        if self.config.tokenizer_model_max_length is not None:
            new_input_embeds = new_input_embeds[:, :self.config.tokenizer_model_max_length]
            new_labels       = new_labels[:, :self.config.tokenizer_model_max_length]

        
        

        # ────────────────────────────── attention mask and position ids ────────────────
        
        attention_mask = (
            torch.arange(new_input_embeds.size(1), device=input_ids.device)
                  .unsqueeze(0)
            < torch.tensor([e.size(0) for e in embeds_list],
                           device=input_ids.device).unsqueeze(1)
        )

        raw_pos = attention_mask.cumsum(dim=1) - 1
        position_ids = raw_pos.masked_fill(~attention_mask, 0).long()

        if not self.training:
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
            images: Optional[PIL.Image.Image] = None,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        is_multi_modal = images is not None and not (
            (
                isinstance(images, list) and (len(images) == 0 or all(i == [] for i in images))
            )
        )


        if inputs_embeds is None and is_multi_modal:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids=input_ids,
                images=images,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
            )

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
    
    def print_device_configuration(self):
        print("*************Device Configuration*********")
        if len(self.get_llm_parameters()) > 0:
            llm_device = set({str(p.device) for p in self.get_llm_parameters()})
            llm_dtype = set({p.dtype for p in self.get_llm_parameters()})
            print(f"LLM Parameters:\t\t\tdevice: {llm_device}\tdtype: {llm_dtype}\tfrozen: {self.is_llm_frozen()}")
        else:
            print("LLM parameters have not been initialized")
        
        if self.get_model().vision_tower is not None:
            vt_device = set({str(p.device) for p in self.get_model().vision_tower.parameters()})
            vt_dtype = set({p.dtype for p in self.get_model().vision_tower.parameters()})
            print(f"Vision Tower Parameters:\tdevice: {vt_device}\tdtype: {vt_dtype}\tfrozen: {self.is_vision_tower_frozen()}")
        else:
            print("Vision tower parameters have not been initialized")

        if self.get_model().mm_projector is not None:
            mm_device = set({str(p.device) for p in self.get_model().mm_projector.parameters()})
            mm_dtype = set({p.dtype for p in self.get_model().mm_projector.parameters()})
            print(f"MM Projector Parameters:\tdevice: {mm_device}\tdtype: {mm_dtype}\tfrozen: {self.is_vision_adapter_frozen()}")
        else:
            print("MM Projector parameters have not been initialized")
        print("******************************************")



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


AutoConfig.register("friday-phi", FridayConfig)
AutoModelForCausalLM.register(FridayConfig, FridayForCausalLM)
