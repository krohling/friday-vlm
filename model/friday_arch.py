import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from .multi_modal_projector.mlp_adapter import MLPAdapter
from .vision_encoder.siglip_encoder import SiglipVisionTowerS2

def pad_and_stack(img_list):
    """
    Args
    ----
    img_list : list[torch.Tensor]
        Each tensor is shape (C, H, W), dtype/ device can differ.

    Returns
    -------
    batch : torch.Tensor
        Shape (B, C, H_max, W_max) on the same device / dtype as the first image.
    """

    print("*************")
    for img in img_list:
        print(img.shape)

    # --- 1. work out the target size ----------------------------------------
    h_max = max(img.shape[1] for img in img_list)
    w_max = max(img.shape[2] for img in img_list)

    if h_max > w_max:
        w_max = h_max
    elif w_max > h_max:
        h_max = w_max

    # --- 2. pad every image to (h_max, w_max) --------------------------------
    padded_imgs = []
    for img in img_list:
        # pad width dimension (left_pad, right_pad) then height (top_pad, bottom_pad)
        pad_w = w_max - img.shape[2]
        pad_h = h_max - img.shape[1]
        # we pad only on the *right* and *bottom*; change the tuple if you want symmetry
        padded = F.pad(img, (0, pad_w,     # width  : (left, right)
                             0, pad_h))    # height : (top,  bottom)
        padded_imgs.append(padded)

    # --- 3. stack into batch --------------------------------------------------
    batch = torch.stack(padded_imgs, dim=0)

    print(batch.shape)

    return batch

class FridayMetaModel(nn.Module):

    def __init__(self, cfg_vision_tower: dict, cfg_vision_adapter: dict):
        super().__init__()
        self.vision_tower = SiglipVisionTowerS2(**cfg_vision_tower)
        self.projector    = MLPAdapter(**cfg_vision_adapter)

    def get_vision_tower(self):
        return self.vision_tower
    
    def encode_images(self, imgs: list) -> torch.Tensor:
        img_tensors = [transforms.ToTensor()(img) for img in imgs]
        imgs = pad_and_stack(img_tensors).to(dtype=torch.float32, device=self.vision_tower.device)
        features = self.vision_tower(imgs)
        return self.projector(features)
    
    def set_vision_projector_requires_grad(self, requires_grad: bool):
        for param in self.projector.parameters():
            param.requires_grad = requires_grad

