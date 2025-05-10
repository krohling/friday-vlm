# test_friday_model_utils.py
#
# Unit‑tests for the utility helpers inside friday.model.FridayModel.
# Heavyweight vision‑tower / adapter classes are monkey‑patched with small
# dummy modules so tests run in <1 s on CPU.
#
# Run with:  pytest -q test_friday_model_utils.py
#
import pytest
import torch
import torch.nn as nn



# --------------------------------------------------------------------------- #
# ---------------------------  Helper: model factory  ----------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def friday_model():
    """Return a FridayModel with very small dims and delay_load=True."""
    from friday.model import FridayModel, FridayConfig

    cfg = FridayConfig(delay_load=True)
    return FridayModel(cfg)


# --------------------------------------------------------------------------- #
# ----------------------------  Actual test cases  -------------------------- #
# --------------------------------------------------------------------------- #
def test_initialize_vision_modules_once(friday_model):
    friday_model.initialize_vision_modules()
    vt1 = friday_model.vision_tower
    proj1 = friday_model.mm_projector

    # Call again – should not create new instances
    friday_model.initialize_vision_modules()
    assert vt1 is friday_model.vision_tower
    assert proj1 is friday_model.mm_projector


def test_compute_image_features_shape(friday_model):
    friday_model.initialize_vision_modules()

    image_size = friday_model.vision_tower.vision_tower.config.image_size
    patch_dim = (image_size // friday_model.vision_tower.vision_tower.config.patch_size) ** 2
    output_dim = friday_model.config.cfg_vision_adapter['output_dim']

    batch, channels, tokens = 2, 3, 4
    imgs = torch.randn(batch, channels, 32, 32)        # spatial dims ignored
    feats = friday_model.compute_image_features(imgs)

    assert feats.shape == (batch, patch_dim, output_dim)          # output_dim = 32


def test_requires_grad_switch(friday_model):
    friday_model.initialize_vision_modules()

    # Freeze adapter
    friday_model.set_vision_adapter_requires_grad(False)
    assert all(not p.requires_grad for p in friday_model.mm_projector.parameters())

    # Unfreeze adapter
    friday_model.set_vision_adapter_requires_grad(True)
    assert all(p.requires_grad for p in friday_model.mm_projector.parameters())

    # Freeze tower
    friday_model.set_vision_tower_requires_grad(False)
    assert all(not p.requires_grad for p in friday_model.vision_tower.parameters())

    # Unfreeze tower
    friday_model.set_vision_tower_requires_grad(True)
    assert all(p.requires_grad for p in friday_model.vision_tower.parameters())


def test_error_uninitialized_adapter(friday_model):
    # Do *not* call initialize_vision_modules()
    with pytest.raises(ValueError):
        friday_model.set_vision_adapter_requires_grad(False)
