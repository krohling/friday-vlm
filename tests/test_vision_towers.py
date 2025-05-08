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

import transformers
# from transformers import SiglipVisionModel, SiglipImageProcessor

# --------------------------------------------------------------------------- #
# -----------------------  Dummy HF‑like building blocks  ------------------- #
# --------------------------------------------------------------------------- #
class DummyProcessor:
    """Mimics `SiglipImageProcessor` but performs no real preprocessing."""
    def __init__(self, side=384):
        self.size = {'height': side, 'width': side}
        self.crop_size = {'height': side, 'width': side}
        self.image_mean = [0.5, 0.5, 0.5]

    @classmethod
    def from_pretrained(cls, *_a, **_kw):
        return cls()

    # returns zeros so the downstream shape logic works
    def __call__(self, _img, return_tensors="pt"):
        side = self.size['height']
        return {'pixel_values': torch.zeros(1, 3, side, side)}


class DummyVisionModel(torch.nn.Module):
    """Mimics `SiglipVisionModel` ‑ returns predictable hidden‑states."""
    def __init__(self, side=384, patch=16, hidden=768, layers=12):
        super().__init__()
        self.config = SimpleNamespace(
            image_size=side,
            patch_size=patch,
            hidden_size=hidden,
        )
        self.dtype = torch.float32

    # ensure `.device` attribute exists
    @property
    def device(self):
        return torch.device("cpu")

    # from_pretrained stub
    @classmethod
    def from_pretrained(cls, *_a, **_kw):
        return cls()

    # no‑op grad switch
    def requires_grad_(self, *_a, **_kw):
        return self

    def forward(self, x, output_hidden_states=True):  # noqa: D401
        # accept (B,C,H,W) or (C,H,W)
        if x.ndim == 3:
            batch = 1
            h, w = x.shape[1], x.shape[2]
        else:
            batch, h, w = x.shape[0], x.shape[2], x.shape[3]

        n_patches = (h // self.config.patch_size) * (w // self.config.patch_size)
        # build dummy hidden state tensor: (B, n_patches, hidden)
        patch_emb = torch.zeros(batch, n_patches, self.config.hidden_size,
                                dtype=x.dtype, device=x.device)
        hidden_states = [patch_emb.clone() for _ in range(12)]
        return SimpleNamespace(hidden_states=hidden_states)


# --------------------------------------------------------------------------- #
# ------------------------  Common monkey‑patch fixture  -------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_siglip(monkeypatch):
    """
    Replace heavyweight Siglip* classes and utilities with lightweight stubs
    *before* the towers are instantiated anywhere.
    """
    import friday.model.vision_tower as vt

    # patch the image processor and vision model
    monkeypatch.setattr(transformers, "SiglipImageProcessor", DummyProcessor, raising=True)
    monkeypatch.setattr(transformers, "SiglipVisionModel", DummyVisionModel, raising=True)

    # simple pad‑and‑stack implementation (all images already equal size)
    def _pad_and_stack(xs):
        return torch.stack(xs)

    monkeypatch.setattr(vt, "pad_and_stack", _pad_and_stack, raising=False)

    # trivial multiscale forward: concat hidden dim for each scale
    def _mock_ms_forward(forward_fn, images, img_sizes, max_split_size):
        outs = [forward_fn(images) for _ in img_sizes]
        return torch.cat(outs, dim=-1)

    monkeypatch.setattr(vt, "multiscale_forward", _mock_ms_forward, raising=False)


# --------------------------------------------------------------------------- #
# -----------------------------  Tower fixtures  ---------------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def tower():
    from friday.model.vision_tower import SiglipVisionTower
    return SiglipVisionTower(model_name_or_path="stub", pad_to_square=True)


@pytest.fixture
def tower_s2():
    from friday.model.vision_tower import SiglipVisionTowerS2
    return SiglipVisionTowerS2(model_name_or_path="stub", s2_scales="256,384")


# --------------------------------------------------------------------------- #
# ----------------------------  Helper functions  --------------------------- #
# --------------------------------------------------------------------------- #
def _dummy_pil(width, height, color=(255, 0, 0)):
    return Image.new("RGB", (width, height), color=color)


# --------------------------------------------------------------------------- #
# ------------------------------  Base tower tests -------------------------- #
# --------------------------------------------------------------------------- #
def test_preprocess_pad_to_square(tower):
    img = _dummy_pil(200, 400)                                 # non‑square
    t = tower.preprocess_images([img])[0]                      # (3,H,W)

    assert t.shape[1] == t.shape[2]                            # square
    assert t.shape[1] == tower.image_processor.size["height"]  # 384


def test_forward_single_tensor_shape(tower):
    img = _dummy_pil(384, 384)
    tensor = tower.preprocess_images([img], pad_and_stack_tensors=False)
    feats = tower(tensor)                                      # (B, P, H)

    expected_patches = (384 // tower.vision_tower.config.patch_size) ** 2
    assert feats.shape == (1, expected_patches,
                           tower.vision_tower.config.hidden_size)


def test_forward_list_batching(tower):
    imgs = [_dummy_pil(384, 384) for _ in range(3)]
    tensors = [tower.preprocess_images([im], pad_and_stack_tensors=False)
               for im in imgs]

    feats_list = tower(tensors)                                # list len 3
    assert isinstance(feats_list, list) and len(feats_list) == 3
    p = (384 // tower.vision_tower.config.patch_size) ** 2
    for feats in feats_list:
        assert feats.shape == (1, p, tower.vision_tower.config.hidden_size)


def test_dtype_and_device_passthrough(tower):
    img = _dummy_pil(384, 384)
    tensor = tower.preprocess_images([img], pad_and_stack_tensors=False)
    feats = tower(tensor)

    assert feats.dtype == tower.dtype
    assert feats.device == tower.device


# --------------------------------------------------------------------------- #
# -------------------------  Multiscale (S2) tower tests -------------------- #
# --------------------------------------------------------------------------- #
def test_multiscale_concat_order(tower_s2):
    side = tower_s2.s2_image_size                              # largest scale
    img_tensor = torch.zeros(3, side, side)

    feats = tower_s2(img_tensor)
    # hidden dim should be hidden_size * n_scales
    assert feats.shape[-1] == tower_s2.vision_tower.config.hidden_size * len(
        tower_s2.s2_scales
    )
    # patch count uses the *largest* scale because dummy forward ignores scale
    p = (side // tower_s2.vision_tower.config.patch_size) ** 2
    assert feats.shape[1] == p


def test_split_tiling_invariance(tower_s2):
    side = tower_s2.s2_image_size
    img_tensor = torch.zeros(3, side, side)

    multi = tower_s2(img_tensor)
    # simulate “no‑tiling” reference
    direct = torch.cat(
        [tower_s2.forward_feature(img_tensor) for _ in tower_s2.s2_scales], dim=-1
    )
    assert torch.allclose(multi, direct, atol=0, rtol=0)
