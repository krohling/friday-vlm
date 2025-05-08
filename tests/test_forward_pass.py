# test_forward_pass.py
#
# “Full‑forward” subsystem tests for friday.model.friday.FridayForCausalLM
# (only the control‑flow; *not* the real Phi or SigLIP weights).
#
# Heavy classes are monkey‑patched with light stubs so every test executes
# in well under a second on CPU and with < 50 MB RAM.
#
# Run:  pytest -q test_forward_pass.py
#
import types
from typing import List

import pytest
import torch
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast

# --------------------------------------------------------------------------- #
# -----------------------------  Dummy building blocks ---------------------- #
# --------------------------------------------------------------------------- #
class DummyVisionTower(torch.nn.Module):
    """Returns a fixed zero tensor; has one fake parameter for .parameters()."""
    def __init__(self, tokens=4, hidden=8):
        super().__init__()
        self.tokens = tokens
        self.hidden = hidden
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, imgs):
        batch = imgs.shape[0] if torch.is_tensor(imgs) else len(imgs)
        return torch.zeros(batch, self.tokens, self.hidden)

    @property
    def device(self):
        return torch.device("cpu")


class DummyAdapter(torch.nn.Module):
    """Small linear projector so grads can flow."""
    def __init__(self, input_dim=8, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.lin = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.lin(x)


# --------------------------------------------------------------------------- #
# -------------------------  Global auto‑patch fixture ---------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_heavy_stuff(monkeypatch):
    """
    *Before* any Friday classes are imported, replace heavy sub‑modules with
    light stubs; patch FridayForCausalLM.__init__ with a lightweight version.
    """
    # --- vision tower / adapter -------------------------------------------------- #
    import friday.model.vision_tower as vt
    import friday.model.vision_adapter as va
    monkeypatch.setattr(vt, "SiglipVisionTower", DummyVisionTower, raising=True)
    monkeypatch.setattr(vt, "SiglipVisionTowerS2", DummyVisionTower, raising=True)
    monkeypatch.setattr(va, "MLPAdapter", DummyAdapter, raising=True)

    # --- lightweight __init__ ---------------------------------------------------- #
    from friday.model.friday import FridayForCausalLM, FridayConfig

    def _light_init(self, config: FridayConfig):
        self.config = config
        self.device = torch.device("cpu")

        hidden = 8
        vocab  = 1000
        config.hidden_size = hidden
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

        # minimal inner model holding vision‑tower & adapter
        class _Inner(torch.nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.embed_tokens = outer.embed_tokens
                self.mm_projector = DummyAdapter(input_dim=8, output_dim=8)
                self.vision_tower = DummyVisionTower()

            def compute_image_features(self, imgs):
                batch = imgs.shape[0]
                return torch.zeros(batch, 1, 8)

            def get_vision_tower(self):
                return self.vision_tower

            def parameters(self, recurse=True):
                return []

        self.model = _Inner(self)

        # expose special token IDs used by helper logic
        self.image_token_id = config.cfg_special_tokens["image_token_id"]
        self.image_start_id = config.cfg_special_tokens["image_start_token_id"]
        self.image_end_id   = config.cfg_special_tokens["image_end_token_id"]

    monkeypatch.setattr(FridayForCausalLM, "__init__", _light_init, raising=True)


# --------------------------------------------------------------------------- #
# ------------------------  Helper factories / utilities -------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def friday():
    from friday.model.friday import FridayForCausalLM, FridayConfig
    cfg = FridayConfig(delay_load=True)
    return FridayForCausalLM(cfg)


def _dummy_pil():
    return Image.new("RGB", (32, 32), color=(0, 0, 0))


# --------------------------------------------------------------------------- #
# 1. forward() with *no images* delegates to Phi3ForCausalLM.forward
# --------------------------------------------------------------------------- #
def test_forward_no_images_equals_phi3(friday, monkeypatch):
    sentinel = object()

    # Mock the superclass forward to capture the call
    import friday.model.friday as friday_mod
    def _fake_phi3_forward(self, *a, **kw):
        return sentinel
    monkeypatch.setattr(friday_mod.Phi3ForCausalLM, "forward", _fake_phi3_forward)

    out = friday.forward(input_ids=torch.tensor([[1, 2, 3]]))
    assert out is sentinel


# --------------------------------------------------------------------------- #
# 2. forward() with images returns a valid CausalLMOutputWithPast
# --------------------------------------------------------------------------- #
def test_forward_with_images_runs(friday, monkeypatch):
    # Replace superclass forward with a dummy that makes logits of correct size
    import friday.model.friday as friday_mod
    def _dummy_phi3_forward(self, input_ids=None, inputs_embeds=None, **_kw):
        batch = inputs_embeds.shape[0]
        seq   = inputs_embeds.shape[1]
        vocab = self.lm_head.out_features
        logits = torch.zeros(batch, seq, vocab, requires_grad=True)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)
    monkeypatch.setattr(friday_mod.Phi3ForCausalLM, "forward", _dummy_phi3_forward)

    img_tok = friday.image_token_id
    ids = torch.tensor([[img_tok, 5]])
    out = friday.forward(
        input_ids=ids,
        images=[_dummy_pil()],          # one PIL image
        labels=None,
    )

    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape[:2] == (1, out.logits.shape[1])  # batch = 1


# --------------------------------------------------------------------------- #
# 3. cache‑position / streaming alignment path
# --------------------------------------------------------------------------- #
def test_cache_position_alignment(friday, monkeypatch):
    # capture the attention_mask that superclass receives
    captured = {}

    import friday.model.friday as friday_mod
    def _spy_phi3_forward(self, *a, **kw):
        captured['mask'] = kw.get("attention_mask")
        return CausalLMOutputWithPast(logits=torch.zeros(1, 1, 10))

    monkeypatch.setattr(friday_mod.Phi3ForCausalLM, "forward", _spy_phi3_forward)

    # dummy past_key_values (len=5 sequence already cached)
    pkv_tensor = torch.zeros(1, 1, 5, 1)
    past_key_values = [(pkv_tensor, pkv_tensor)]

    img_tok = friday.image_token_id
    friday.forward(
        input_ids=torch.tensor([[img_tok]]),
        images=[_dummy_pil()],
        past_key_values=past_key_values,
    )

    new_mask = captured['mask']
    # Stream step should expand to old_len + 1 == 6
    assert new_mask.shape[1] == 6
    # last position is the newly appended position (mask==1)
    assert new_mask[0, -1] == 1


# --------------------------------------------------------------------------- #
# 4. Gradients flow only through adapter when everything else frozen
# --------------------------------------------------------------------------- #
def test_gradients_flow_to_adapter_only(friday, monkeypatch):
    # Dummy superclass forward with differentiable logits
    import friday.model.friday as friday_mod
    def _grad_phi3_forward(self, input_ids=None, inputs_embeds=None, **kw):
        logits = self.lm_head(inputs_embeds)            # uses lm_head params
        return CausalLMOutputWithPast(logits=logits)

    monkeypatch.setattr(friday_mod.Phi3ForCausalLM, "forward", _grad_phi3_forward)

    # Freeze language embeddings & vision tower; keep adapter trainable
    for p in friday.embed_tokens.parameters():
        p.requires_grad_(False)
    for p in friday.lm_head.parameters():
        p.requires_grad_(False)
    for p in friday.model.vision_tower.parameters():
        p.requires_grad_(False)
    for p in friday.model.mm_projector.parameters():
        p.requires_grad_(True)

    # Forward + backward with a dummy loss
    img_tok = friday.image_token_id
    out = friday.forward(
        input_ids=torch.tensor([[img_tok]]),
        images=[_dummy_pil()],
        labels=None,
    )
    loss = out.logits.mean()
    loss.backward()

    # Adapter should have grads; others should not
    assert all(p.grad is not None for p in friday.model.mm_projector.parameters())
    assert all(p.grad is None for p in friday.embed_tokens.parameters())
    assert all(p.grad is None for p in friday.lm_head.parameters())
    assert all(p.grad is None for p in friday.model.vision_tower.parameters())
