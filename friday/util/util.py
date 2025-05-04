import torch
import torch.nn.functional as F

from PIL import Image

def get_module_device(module: torch.nn.Module) -> torch.device:
    """Return the device where the module’s first parameter/buffer lives.

    If the module has no parameters or buffers (rare), default to CPU."""
    # parameters() is an iterator, so we only pull the first
    try:
        return next(module.parameters()).device
    except StopIteration:
        # e.g. an empty container module
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def pad_and_stack(img_list, pad_value=0.0):
    """
    img_list : list[Tensor]  each (C, H, W) already *normalised*
    pad_value: float or tuple/list of 3 floats (one per channel)
               Use 0.0 if your processor has already centred to mean 0.
    Returns
    -------
    batch : Tensor  (B, C, H_max, W_max)
    """

    # 1. target square size ---------------------------------------------------
    h_max = max(t.shape[1] for t in img_list)
    w_max = max(t.shape[2] for t in img_list)
    H, W  = max(h_max, w_max), max(h_max, w_max)

    # 2. create padded copies -------------------------------------------------
    padded = []
    for img in img_list:
        c, h, w = img.shape
        canvas   = img.new_full((c, H, W), pad_value)     # filled with mean/zeros
        canvas[:, :h, :w] = img                    # top-left corner
        padded.append(canvas)

    return torch.stack(padded, 0)                  # (B,C,H,W)
