import torch
import torch.nn as nn
import torch.nn.functional as F

import PIL.Image
from typing import List
from friday.util import expand2square, pad_and_stack

from transformers import AutoModel, AutoImageProcessor
from friday.util.s2wrapper import forward as multiscale_forward


class FastVitVisionTower(nn.Module):
    def __init__(self, pretrained_model_name_or_path, model_params={}, pad_to_square=True, **kwargs):
        super().__init__()

        self.is_loaded = False
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_params = model_params
        self.pad_to_square = pad_to_square
        self.load_model()

    @property
    def output_dim(self):
        return self.vision_tower.config.embed_dim if self.vision_tower else None
    
    def load_model(self):
        if self.is_loaded:
            return
        self.image_processor = AutoImageProcessor.from_pretrained(self.pretrained_model_name_or_path)
        self.image_processor.crop_size = self.image_processor.size
        self.vision_tower = AutoModel.from_pretrained(
            self.pretrained_model_name_or_path,
            **self.model_params,
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
    
    def preprocess_images(self, imgs: List[PIL.Image.Image], pad_and_stack_tensors=True) -> torch.Tensor:
        img_mean = tuple(int(x * 255) for x in self.image_processor.image_mean)
        if self.pad_to_square:
            imgs = [expand2square(img, img_mean) for img in imgs]
        
        imgs = [self.image_processor(img, do_resize=True, do_center_crop=False, return_tensors="pt")['pixel_values'][0] for img in imgs]
        

        if pad_and_stack_tensors:
            imgs = pad_and_stack(imgs, pad_value=0.0)
            imgs = imgs.to(dtype=torch.float32, device=self.device)
        
        return imgs

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
            )

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.embed_dim, device=self.device, dtype=self.dtype)

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
        return self.config.embed_dim

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class FastVitVisionTowerS2(FastVitVisionTower):
    def __init__(self, pretrained_model_name_or_path, s2_scales, model_params={}, **kwargs):
        self.s2_scales = list(map(int, s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(pretrained_model_name_or_path, model_params)

        self.multiscale_forward = multiscale_forward
    
    @property
    def output_dim(self):
        return (2*self.vision_tower.config.embed_dim) if self.vision_tower else None

    def load_model(self):
        if self.is_loaded:
            return
        
        super().load_model()
        self.image_processor.size = self.image_processor.crop_size = {
            "height": self.s2_image_size,
            "width":  self.s2_image_size
        }

    def forward_feature(self, images):
        image_size = self.vision_tower.config.image_size
        if images.shape[2] != image_size or images.shape[3] != image_size:
            images = F.interpolate(
                images,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )    

        return self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
        )

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature, 
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales, 
                    max_split_size=self.s2_split_size
                )
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature, 
                images, 
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.embed_dim * len(self.s2_scales)
