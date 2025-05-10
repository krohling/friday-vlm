# test_forward_pass.py
#
# “Full‑forward” subsystem tests for friday.model.FridayForCausalLM
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
from torchvision import transforms
from transformers.modeling_outputs import CausalLMOutputWithPast


# --------------------------------------------------------------------------- #
# ------------------------  Helper factories / utilities -------------------- #
# --------------------------------------------------------------------------- #
@pytest.fixture
def friday():
    from friday.model import FridayForCausalLM, FridayConfig

    cfg = FridayConfig(delay_load=True)
    model = FridayForCausalLM(cfg)
    model.initialize_vision_modules()
    return model


def _dummy_pil():
    return Image.new("RGB", (32, 32), color=(0, 0, 0))


# --------------------------------------------------------------------------- #
# 1. forward() with *no images* delegates to Phi3ForCausalLM.forward
# --------------------------------------------------------------------------- #
def test_forward_no_images_equals_phi3(friday, monkeypatch):
    sentinel = object()

    # Mock the superclass forward to capture the call
    import friday.model.language_model.phi4 as lm_mod
    def _fake_phi3_forward(self, *a, **kw):
        return sentinel
    monkeypatch.setattr(lm_mod.Phi3ForCausalLM, "forward", _fake_phi3_forward)

    out = friday.forward(input_ids=torch.tensor([[1, 2, 3]]))
    assert out is sentinel


# --------------------------------------------------------------------------- #
# 2. forward() with images returns a valid CausalLMOutputWithPast
# --------------------------------------------------------------------------- #
def test_forward_with_images_runs(friday, monkeypatch):
    # Replace superclass forward with a dummy that makes logits of correct size
    import friday.model.language_model.phi4 as lm_mod
    def _dummy_phi3_forward(self, input_ids=None, inputs_embeds=None, **_kw):
        batch = inputs_embeds.shape[0]
        seq   = inputs_embeds.shape[1]
        vocab = self.lm_head.out_features
        logits = torch.zeros(batch, seq, vocab, requires_grad=True)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)
    monkeypatch.setattr(lm_mod.Phi3ForCausalLM, "forward", _dummy_phi3_forward)

    img_tok = friday.image_token_id
    ids = torch.tensor([[img_tok]])
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

    import friday.model.language_model.phi4 as lm_mod
    def _spy_phi3_forward(self, *a, **kw):
        captured['mask'] = kw.get("attention_mask")
        return CausalLMOutputWithPast(logits=torch.zeros(1, 1, 10))

    monkeypatch.setattr(lm_mod.Phi3ForCausalLM, "forward", _spy_phi3_forward)

    # dummy past_key_values (len=5 sequence already cached)
    pkv_tensor = torch.zeros(1, 1, 5, 1)
    prev_attention_mask = torch.zeros(1, 5)
    past_key_values = [(pkv_tensor, pkv_tensor)]

    img_tok = friday.image_token_id
    friday.forward(
        input_ids=torch.tensor([[img_tok]]),
        images=[_dummy_pil()],
        past_key_values=past_key_values,
        attention_mask=prev_attention_mask,
    )

    new_mask = captured['mask']
    # Stream step should expand to old_len + 1 == 6
    assert new_mask.shape[1] == 6
    # last position is the newly appended position (mask==1)
    assert new_mask[0, -1] == 1


# --------------------------------------------------------------------------- #
# 4. Gradients flow only through adapter when everything else frozen
# --------------------------------------------------------------------------- #
def test_gradients_flow_to_adapter_only(friday):
    # Freeze language embeddings & vision tower; keep adapter trainable
    friday.set_language_model_requires_grad(False)
    friday.set_vision_tower_requires_grad(False)
    friday.set_vision_adapter_requires_grad(True)

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
    for n, p in friday.named_parameters():
        if "mm_projector" in n:
            assert p.grad is not None
        else:
            assert p.grad is None
