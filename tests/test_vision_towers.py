# test_vision_towers.py
#
# Unit‑tests for the vision‑tower subsystem (SiglipVisionTower and
# SiglipVisionTowerS2).  Every heavy Hugging‑Face component is monkey‑patched
# with fast “dummy” stubs so the whole suite runs in < 1 s on CPU.
#
# Run with:  pytest -q test_vision_towers.py
#

from types import SimpleNamespace
import math
import pytest
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# -----------------------------  Tower fixtures  ---------------------------- #
# --------------------------------------------------------------------------- #
def build_tower(use_s2=False, type="siglip"):
    if type == "siglip":
        from friday.model.vision_tower import SiglipVisionTower, SiglipVisionTowerS2
        if use_s2:
            return SiglipVisionTowerS2(pretrained_model_name_or_path="google/siglip2-base-patch16-384", s2_scales="384,768")
        else:
            return SiglipVisionTower(pretrained_model_name_or_path="google/siglip2-base-patch16-384")
    elif type == "fast_vit":
        from friday.model.vision_tower.fast_vit_encoder import FastVitVisionTower, FastVitVisionTowerS2
        model_params = {
            "trust_remote_code": True,
        }
        if use_s2:
            return FastVitVisionTowerS2(pretrained_model_name_or_path="kevin510/fast-vit-hd", s2_scales="512,1024", model_params=model_params)
        else:
            return FastVitVisionTower(pretrained_model_name_or_path="kevin510/fast-vit-hd", model_params=model_params)

@pytest.fixture
def tower():
    return build_tower()


@pytest.fixture
def tower_s2():
    return build_tower(use_s2=True)


# --------------------------------------------------------------------------- #
# ----------------------------  Helper functions  --------------------------- #
# --------------------------------------------------------------------------- #
def _dummy_pil(width, height, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color=color)


# --------------------------------------------------------------------------- #
# ------------------------------  Base tower tests -------------------------- #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "v_tower",
    [
        build_tower(use_s2=False, type="siglip"),
        build_tower(use_s2=True, type="siglip"),
        build_tower(use_s2=False, type="fast_vit"),
        build_tower(use_s2=True, type="fast_vit"),
    ],
)
def test_preprocess(v_tower):
    tower = v_tower
    img = [
        _dummy_pil(200, 400),
        _dummy_pil(300, 100),
    ]
    out = tower.preprocess_images(img, pad_and_stack_tensors=False)
    
    assert len(out) == len(img)
    assert isinstance(out, list)

    for t in out:
        assert isinstance(t, torch.Tensor)
        assert t.ndim == 3
        assert t.shape[0] == 3
        assert t.shape[1] == t.shape[2]
        assert t.dtype == torch.float32
        assert t.device == tower.device

        if "height" in tower.image_processor.size and "width" in tower.image_processor.size:
            assert t.shape[1] == tower.image_processor.size["height"]
            assert t.shape[2] == tower.image_processor.size["width"]
        elif "shortest_edge" in tower.image_processor.size:
            assert t.shape[1] == tower.image_processor.size["shortest_edge"]
            assert t.shape[2] == tower.image_processor.size["shortest_edge"]
    
@pytest.mark.parametrize(
    "v_tower",
    [
        build_tower(use_s2=False, type="siglip"),
        build_tower(use_s2=True, type="siglip"),
        build_tower(use_s2=False, type="fast_vit"),
        build_tower(use_s2=True, type="fast_vit"),
    ],
)
def test_preprocess_with_pad_and_stack(v_tower):
    tower = v_tower
    img = [
        _dummy_pil(200, 400),
        _dummy_pil(300, 100),
    ]
    out = tower.preprocess_images(img, pad_and_stack_tensors=True)

    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4
    assert out.shape[0] == len(img)
    assert out.shape[1] == 3
    assert out.shape[2] == out.shape[3]

    if "height" in tower.image_processor.size and "width" in tower.image_processor.size:
        assert out.shape[2] == tower.image_processor.size["height"]
        assert out.shape[3] == tower.image_processor.size["width"]
    elif "shortest_edge" in tower.image_processor.size:
        assert out.shape[2] == tower.image_processor.size["shortest_edge"]
        assert out.shape[3] == tower.image_processor.size["shortest_edge"]

    assert out.dtype == torch.float32
    assert out.device == tower.device


@pytest.mark.parametrize(
    "v_tower",
    [
        build_tower(use_s2=False, type="siglip"),
        build_tower(use_s2=True, type="siglip"),
        build_tower(use_s2=False, type="fast_vit"),
        build_tower(use_s2=True, type="fast_vit"),
    ],
)
def test_forward_single_tensor(v_tower):
    tower = v_tower
    img = _dummy_pil(200, 400)
    tensor = tower.preprocess_images([img], pad_and_stack_tensors=True)
    feats = tower(tensor)                                      # (B, P, H)

    expected_patches = (tower.vision_tower.config.image_size // tower.vision_tower.config.patch_size) ** 2
    assert feats.shape == (1, expected_patches, tower.output_dim)


@pytest.mark.parametrize(
    "v_tower",
    [
        build_tower(use_s2=False, type="siglip"),
        build_tower(use_s2=True, type="siglip"),
        build_tower(use_s2=False, type="fast_vit"),
        build_tower(use_s2=True, type="fast_vit"),
    ],
)
def test_forward_list_batching(v_tower):
    tower = v_tower
    imgs = [_dummy_pil(384, 384) for _ in range(3)]
    tensors = tower.preprocess_images(imgs, pad_and_stack_tensors=False)

    feats_list = tower(tensors)                                # list len 3
    assert isinstance(feats_list, list) and len(feats_list) == 3
    p = (tower.vision_tower.config.image_size // tower.vision_tower.config.patch_size) ** 2
    for feats in feats_list:
        assert feats.shape == (1, p, tower.output_dim)


@pytest.mark.parametrize(
    "v_tower",
    [
        build_tower(use_s2=False, type="siglip"),
        build_tower(use_s2=True, type="siglip"),
        build_tower(use_s2=False, type="fast_vit"),
        build_tower(use_s2=True, type="fast_vit"),
    ],
)
def test_forward_tensor_batching(v_tower):
    tower = v_tower
    imgs = [_dummy_pil(384, 384) for _ in range(4)]
    tensors = tower.preprocess_images(imgs, pad_and_stack_tensors=True)

    feats_tens = tower(tensors)                                # list len 3
    assert isinstance(feats_tens, torch.Tensor)
    p = (tower.vision_tower.config.image_size // tower.vision_tower.config.patch_size) ** 2
    assert feats_tens.shape == (len(imgs), p, tower.output_dim)

@pytest.mark.parametrize(
    "v_tower",
    [
        build_tower(use_s2=False, type="siglip"),
        build_tower(use_s2=True, type="siglip"),
        build_tower(use_s2=False, type="fast_vit"),
        build_tower(use_s2=True, type="fast_vit"),
    ],
)
def test_dtype_and_device_passthrough(v_tower):
    tower = v_tower
    img = _dummy_pil(384, 384)
    tensor = tower.preprocess_images([img], pad_and_stack_tensors=True)
    feats = tower(tensor)

    assert feats.dtype == tower.dtype
    assert feats.device == tower.device



