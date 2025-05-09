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
# ---------------------------  Dummy building blocks  ----------------------- #
# --------------------------------------------------------------------------- #
class DummyVisionTower(nn.Module):
    """Extremely small stand‑in for SiglipVisionTower(S2)."""
    def __init__(self, output_tokens=4, hidden_size=16, **_kw):
        super().__init__()
        self.output_tokens = output_tokens
        self.hidden_size = hidden_size
        # single parameter so `.parameters()` is not empty
        self.weight = nn.Parameter(torch.zeros(1))

    # called by FridayModel.initialize_vision_modules()
    def load_model(self):
        pass

    def forward(self, imgs):
        batch = imgs.shape[0]
        return torch.zeros(batch, self.output_tokens, self.hidden_size,
                           dtype=imgs.dtype, device=imgs.device)

    @property
    def device(self):
        return torch.device("cpu")


class DummyProjector(nn.Module):
    """Simple linear projection with attribute `output_dim`."""
    def __init__(self, input_dim, hidden_dim, output_dim, **_kw):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


# --------------------------------------------------------------------------- #
# --------------------  Automatic monkey‑patch for every test --------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_friday_modules(monkeypatch):
    """
    Replace Siglip towers & MLPAdapter with dummy modules **before** the model
    is instantiated so that `initialize_vision_modules()` constructs dummies.
    """
    import friday.model.vision_tower as vt
    import friday.model.vision_adapter as va

    # monkey‑patch both tower classes to DummyVisionTower
    monkeypatch.setattr(vt, "SiglipVisionTower", DummyVisionTower, raising=True)
    monkeypatch.setattr(vt, "SiglipVisionTowerS2", DummyVisionTower, raising=True)
    monkeypatch.setattr(va, "MLPAdapter", DummyProjector, raising=True)


# --------------------------------------------------------------------------- #
# ---------------------------  Helper: model factory  ----------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def friday_model():
    """Return a FridayModel with very small dims and delay_load=True."""
    from friday.model import FridayModel, FridayConfig

    cfg = FridayConfig(delay_load=True)
    cfg.cfg_vision_adapter.update(dict(input_dim=16, hidden_dim=8, output_dim=32))
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

    batch, channels, tokens = 2, 3, 4
    imgs = torch.randn(batch, channels, 32, 32)        # spatial dims ignored
    feats = friday_model.compute_image_features(imgs)

    assert feats.shape == (batch, tokens, 32)          # output_dim = 32


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


def test_error_uninitialized_adapter(friday_model):
    # Do *not* call initialize_vision_modules()
    with pytest.raises(ValueError):
        friday_model.set_vision_adapter_requires_grad(False)
