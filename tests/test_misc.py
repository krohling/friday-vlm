# test_errors_and_utils.py
#
# Tests for
#   • “Error‑handling edge‑cases” that should raise
#   • “Miscellaneous utility” helpers (build_tokenizer)
#
# Every heavy external dependency is patched with a very small stub so the
# suite runs quickly on CPU with no network access.
#
# Run with:  pytest -q test_errors_and_utils.py
#
import types
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------- #
# ----------------------  Shared lightweight helper classes ------------------ #
# ---------------------------------------------------------------------------- #
class DummyVisionTower:
    def preprocess_images(self, imgs, pad_and_stack_tensors=True):
        # trivial tensor per image
        return torch.zeros(len(imgs), 3, 32, 32)


class DummyProjector(torch.nn.Module):
    output_dim = 4

    def forward(self, x):
        return torch.zeros(*x.shape[:-1], self.output_dim)


class DummyTokenizerHF:
    """Minimal Hugging‑Face‑like tokenizer used by build_tokenizer tests."""
    def __init__(self):
        # Only eos exists at construction time
        self.vocab = {"<|eos|>": 2}
        self._next_id = 3

        self.pad_token_id = None
        self.eos_token_id = 2
        self.padding_side = "right"

    # HF AutoTokenizer.from_pretrained replacement -------------------------- #
    @classmethod
    def from_pretrained(cls, *_a, **kw):
        return cls()

    # token helpers --------------------------------------------------------- #
    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok, None)

    def add_tokens(self, toks, special_tokens=False):
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._next_id
                self._next_id += 1
        return len(toks)

    # HF attribute used by build_tokenizer
    @property
    def eos_token(self):
        return "<|eos|>"

    @property
    def pad_token(self):
        return self._pad_tok

    @pad_token.setter
    def pad_token(self, tok):
        self._pad_tok = tok
        self.pad_token_id = self.convert_tokens_to_ids(tok)


def _dummy_pil():
    return Image.new("RGB", (32, 32))


# ---------------------------------------------------------------------------- #
# -----------------------  Pytest‑wide monkey‑patching ----------------------- #
# ---------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_everything(monkeypatch):
    """
    Patch *before* the modules we test are imported.
    """
    # --- Friday constants --------------------------------------------------- #
    import friday.constants as const
    monkeypatch.setattr(const, "IMAGE_TOKEN", "<image>", raising=True)
    monkeypatch.setattr(const, "IMG_START_TOKEN", "<img_start>", raising=True)
    monkeypatch.setattr(const, "IMG_END_TOKEN",   "<img_end>",   raising=True)
    monkeypatch.setattr(const, "IGNORE_INDEX", -100,             raising=True)

    # --- Friday util device helper (for print_device_configuration) -------- #
    import friday.util as futil
    monkeypatch.setattr(futil, "get_module_device",
                        lambda m: next(m.parameters()).device
                        if any(True for _ in m.parameters()) else torch.device("cpu"),
                        raising=True)

    # --- Light stubs for vision tower & adapter ---------------------------- #
    import friday.model.vision_tower as vt
    import friday.model.vision_adapter as va
    monkeypatch.setattr(vt, "SiglipVisionTower", DummyVisionTower, raising=True)
    monkeypatch.setattr(vt, "SiglipVisionTowerS2", DummyVisionTower, raising=True)
    monkeypatch.setattr(va, "MLPAdapter", DummyProjector, raising=True)

    # --- Patch HF AutoTokenizer with our dummy ----------------------------- #
    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", types.SimpleNamespace(
        from_pretrained=DummyTokenizerHF.from_pretrained
    ), raising=True)

    yield


# ---------------------------------------------------------------------------- #
# -------------------------  Lightweight Friday object ---------------------- #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def friday(monkeypatch):
    """FridayForCausalLM with helper logic intact but zero heavy weights."""
    from friday.model import FridayForCausalLM, FridayConfig

    # patch FridayForCausalLM.__init__ to be minimal ------------------------- #
    def _light_init(self, config: FridayConfig):
        torch.nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = torch.nn.Embedding(100, 8)
        self.image_token_id = config.cfg_special_tokens["image_token_id"]
        self.image_start_id = config.cfg_special_tokens["image_start_token_id"]
        self.image_end_id   = config.cfg_special_tokens["image_end_token_id"]
        self.to(torch.device("cpu"))

        class _Inner(torch.nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.embed_tokens = outer.embed_tokens
                self.mm_projector = DummyProjector()
                self.vision_tower = DummyVisionTower()

            def compute_image_features(self, imgs):
                batch = imgs.shape[0]
                return torch.zeros(batch, 1, 4)

            def get_vision_tower(self):
                return self.vision_tower

            def parameters(self, recurse=True):
                return []

        self.model = _Inner(self)
        self.lm_head = torch.nn.Linear(8, 100, bias=False)

    monkeypatch.setattr("friday.model.FridayForCausalLM.__init__", _light_init)

    return FridayForCausalLM(FridayConfig(delay_load=True))


# ---------------------------------------------------------------------------- #
# --------------------------  Error‑handling tests -------------------------- #
# ---------------------------------------------------------------------------- #
def test_prepare_inputs_empty_images_raises(friday):
    ids = torch.tensor([[friday.image_token_id]])
    with pytest.raises(Exception):         # IndexError *or* ValueError acceptable
        friday.prepare_inputs_for_multimodal(
            input_ids=ids,
            images=[],                     # empty list triggers error path
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
        )


def test_prepare_inputs_unsupported_type_raises(friday):
    ids = torch.tensor([[friday.image_token_id]])
    with pytest.raises(ValueError):
        friday.prepare_inputs_for_multimodal(
            input_ids=ids,
            images=123,                    # unsupported type
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
        )


# ---------------------------------------------------------------------------- #
# -----------------------  Miscellaneous utility tests ---------------------- #
# ---------------------------------------------------------------------------- #
def test_build_tokenizer_adds_specials(monkeypatch):
    # Import *after* AutoTokenizer patch
    from friday.model import build_tokenizer

    tok, specials = build_tokenizer("kevin510/friday")

    expected = {"<image>", "<img_start>", "<img_end>"}
    # specials mapping keys may differ; we care about IDs presence & uniqueness
    assert set(specials.values()) == {tok.convert_tokens_to_ids(t) for t in expected}
    # pad_token set equal to eos_token
    assert tok.pad_token_id == tok.eos_token_id
