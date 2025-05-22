import torch
import torch.nn.functional as F

from PIL import Image


# def mask_token_segment(start_token, end_token, input_ids, mask_value=-100):
#     is_start_token = input_ids == start_token
#     cum_start = torch.cumsum(is_start_token, dim=0)

#     is_end_token = (input_ids == end_token)

#     cum_end = torch.cumsum(is_end_token, dim=0)

#     print("mask")
#     print(cum_end.masked_fill(cum_end > cum_start, 0))

#     is_masked = (cum_start > cum_end)

#     print(f"is_start_token: {is_start_token}")
#     print(f"is_end_token: {is_end_token}")
#     print(f"cum_start: {cum_start}")
#     print(f"cum_end: {cum_end}")
#     print(f"is_masked: {is_masked}")

#     result = input_ids.clone()
#     result[is_masked] = mask_value
#     return result

import torch

def mask_token_segment(
               start_id: int,
               end_id: int,
               input_ids: torch.Tensor,
               fill_value: int = -100):
    """
    Replace *every* token from each `start_id` **through** its matching `end_id`
    (boundaries included) with `fill_value`.  Any spans that start with some
    other token are left untouched.

    Works on CUDA, TorchScript, batched via vmap, etc.—no Python loops.
    """
    if input_ids.dim() != 1:
        raise ValueError("`input_ids` must be 1-D")

    device = input_ids.device
    n       = input_ids.size(0)

    # where the *target* start-tokens and end-tokens sit
    start_pos = (input_ids == start_id).nonzero(as_tuple=True)[0]      # ascending
    end_pos   = (input_ids == end_id).nonzero(as_tuple=True)[0]        # ascending

    if start_pos.numel() == 0:
        return input_ids.clone()

    # ── pair every start with the first end that comes *after* it ────────────────
    # searchsorted gives the insertion index into the (sorted) end positions
    idx_in_end = torch.searchsorted(end_pos, start_pos, right=False)

    have_match = idx_in_end < end_pos.size(0)                # safety: drop unmatched
    start_pos  = start_pos[have_match]
    end_pos    = end_pos[idx_in_end[have_match]]

    # (rare) guard against pathological orderings
    keep = end_pos > start_pos
    start_pos, end_pos = start_pos[keep], end_pos[keep]

    if start_pos.numel() == 0:
        return input_ids

    # ── differential “scan-line” trick to build the span mask in O(N) ───────────
    # +1 at each start index, -1 at the element *after* each end
    delta = torch.zeros(n + 1, dtype=torch.int8, device=device)
    delta[start_pos]        += 1
    delta[end_pos + 1]      -= 1          # +1 is safe because delta is length n+1

    inside = torch.cumsum(delta[:-1], dim=0) > 0   # boolean mask, incl. boundaries

    # ── apply ────────────────────────────────────────────────────────────────────
    out = input_ids.clone()
    out[inside] = fill_value
    return out



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.util.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


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
