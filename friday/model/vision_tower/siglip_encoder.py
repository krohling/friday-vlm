import torch
import torch.nn as nn

import PIL.Image
from typing import List
from friday.util import expand2square, pad_and_stack

from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
from friday.util.s2wrapper import forward as multiscale_forward


class SiglipVisionTower(nn.Module):
    def __init__(self, model_name_or_path, model_params={}, pad_to_square=True, **kwargs):
        super().__init__()

        self.is_loaded = False
        self.model_name_or_path = model_name_or_path
        self.model_params = model_params
        self.pad_to_square = pad_to_square
        self.select_layer = -2
        self.load_model()

    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.model_name_or_path)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.model_name_or_path,
            **self.model_params,
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def preprocess_images(self, imgs: List[PIL.Image.Image], pad_and_stack_tensors=True) -> torch.Tensor:
        img_mean = tuple(int(x * 255) for x in self.image_processor.image_mean)
        if self.pad_to_square:
            imgs = [expand2square(img, img_mean) for img in imgs]
        imgs = [self.image_processor(img, return_tensors="pt")['pixel_values'][0] for img in imgs]

        if pad_and_stack_tensors:
            imgs = pad_and_stack(imgs)
            imgs = imgs.to(dtype=torch.float32, device=self.device)
        
        return imgs

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]

        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class SiglipVisionTowerS2(SiglipVisionTower):
    def __init__(self, model_name_or_path, s2_scales, model_params={}, **kwargs):
        self.s2_scales = list(map(int, s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(model_name_or_path, model_params)

        self.multiscale_forward = multiscale_forward

        self.image_processor.size['height'] = self.image_processor.size['width'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
    
    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = SiglipImageProcessor.from_pretrained(self.model_name_or_path)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = SiglipVisionModel.from_pretrained(
            self.model_name_or_path,
            **self.model_params,
        )
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['height'] = self.image_processor.size['width'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        print(f"images: {images.shape}")
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0),
                                                        img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales,
                                                     max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
