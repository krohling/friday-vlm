# test_data_pipeline.py
#
# Unit tests for the data‑pipeline utilities:
#   • preprocess_for_pretraining
#   • PretrainingDataset
#   • FridayCollator
#
# Heavy external dependencies (real tokenizer, SigLIP tower, etc.) are replaced
# with fast stubs so the whole file executes in < 1 s on CPU.
#
# Run with:  pytest -q test_data_pipeline.py
#
import json
import os
from types import SimpleNamespace
from pathlib import Path
from transformers import AutoTokenizer

import pytest
import torch
from PIL import Image

from util import test_images_paths

# ---------------------------------------------------------------------------- #
# ---------------------------  Lightweight stubs ----------------------------- #
# ---------------------------------------------------------------------------- #
# class DummyVisionTower:
#     """Provides the single method used in preprocess_for_pretraining."""
#     def preprocess_images(self, imgs, pad_and_stack_tensors=False):
#         # returns a (3,32,32) zero tensor for each image
#         if pad_and_stack_tensors:
#             return torch.zeros(len(imgs), 3, 32, 32)
#         return [torch.zeros(3, 32, 32) for _ in imgs]
    
#     @property
#     def num_patches(self):
#         return 32


# class DummyTokenizer:
#     """Minimal tokenizer fulfilling Pretraining* needs."""
#     def __init__(self, pad_id=0, eos_id=2, img_id=1):
#         self.pad_token_id = pad_id
#         self.eos_token_id = eos_id
#         self.model_max_length = 128

#         self.vocab = {"<image>": img_id}
#         self._next = 100  # next fresh id

#     # ─── tokenization ────────────────────────────────────────────────────────
#     def __call__(self, text, return_tensors="pt", **_kw):
#         ids = [self._tok2id(tok) for tok in text.split()]
#         return SimpleNamespace(input_ids=torch.tensor([ids]))

#     def _tok2id(self, tok):
#         if tok not in self.vocab:
#             self.vocab[tok] = self._next
#             self._next += 1
#         return self.vocab[tok]

def build_tower(use_s2=True):
    from friday.model.vision_tower import SiglipVisionTower, SiglipVisionTowerS2
    if use_s2:
        return SiglipVisionTowerS2(pretrained_model_name_or_path="google/siglip2-base-patch16-384", s2_scales="384,768")
    else:
        return SiglipVisionTower(pretrained_model_name_or_path="google/siglip2-base-patch16-384")

@pytest.fixture
def vision_tower():
    return build_tower()

def build_tokenizer():
    return AutoTokenizer.from_pretrained("kevin510/friday")

@pytest.fixture
def tokenizer():
    return build_tokenizer()


# ---------------------------------------------------------------------------- #
# --------------------------  PyTest‑wide monkey‑patch ----------------------- #
# ---------------------------------------------------------------------------- #
# @pytest.fixture(autouse=True)
# def patch_modules(monkeypatch):
#     """
#     *Every* test uses the same patched constants & stubs.
#     """
#     # Patch constants used by pipeline
#     import friday.constants as const
#     # monkeypatch.setattr(const, "IMAGE_TOKEN", "<image>", raising=True)
#     monkeypatch.setattr(const, "PAD_FOR_EOS", -123,      raising=True)
#     monkeypatch.setattr(const, "IGNORE_INDEX", -100,     raising=True)

#     # SPECIAL_TOKENS lives in friday.model namespace
#     import friday.model as fm
#     fm.SPECIAL_TOKENS = {"image_token_id": 1}

#     yield


# ---------------------------------------------------------------------------- #
# -----------------------  Helpers for temporary sample data ----------------- #
# ---------------------------------------------------------------------------- #

def _make_sample(img_file):
    return {
        "image": str(img_file),
        "caption": "A red square.",
    }

# ---------------------------------------------------------------------------- #
# ---------------------------  7.1 preprocess_for_pretraining ---------------- #
# ---------------------------------------------------------------------------- #
def test_prompt_construction(test_images_paths, tokenizer, vision_tower):
    from friday.data import preprocess_for_pretraining

    sample = _make_sample(test_images_paths[0])

    out = preprocess_for_pretraining(sample, str(test_images_paths[0].parent), vision_tower, tokenizer)

    img_tok_id = tokenizer.vocab["<image>"]
    # first token(s) equal <image>
    assert (out["input_ids"][:1] == img_tok_id).all()


def test_label_masking(test_images_paths, tokenizer, vision_tower):
    from friday.data import preprocess_for_pretraining

    sample = _make_sample(test_images_paths[0])

    out = preprocess_for_pretraining(sample, str(test_images_paths[0].parent), vision_tower, tokenizer)

    img_tok_id = tokenizer.vocab["<image>"]
    mask_positions = out["input_ids"] == img_tok_id

    assert torch.all(out["labels"][mask_positions] == -100)   # IGNORE_INDEX


def test_missing_image_assert(tmp_path, tokenizer, vision_tower):
    from friday.data import preprocess_for_pretraining

    sample = {
        "image": None,
        "conversations": [{"from": "gpt", "value": "hi"}],
    }

    with pytest.raises(AssertionError):
        preprocess_for_pretraining(sample, str(tmp_path), vision_tower, tokenizer)


# ---------------------------------------------------------------------------- #
# -----------------------------  7.2 PretrainingDataset ---------------------- #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def json_dataset(tmp_path: Path, test_images_paths):
    """Write a JSON file with two samples and return its path."""
    img0, img1 = test_images_paths
    samples = [_make_sample(img0), _make_sample(img1)]
    json_path = tmp_path / "samples.json"
    json_path.write_text(json.dumps(samples))
    return json_path, img0.parent, len(samples)


def test_len_matches_json(json_dataset, tokenizer, vision_tower):
    from friday.data import PretrainingDataset

    json_path, img_dir, N = json_dataset
    ds = PretrainingDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    assert len(ds) == N


def test_get_item_keys_types(json_dataset, tokenizer, vision_tower):
    from friday.data import PretrainingDataset

    json_path, img_dir, _ = json_dataset
    ds = PretrainingDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "labels", "image"}
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert isinstance(item["image"], list) and torch.is_tensor(item["image"][0])


# ---------------------------------------------------------------------------- #
# -----------------------------  7.3 FridayCollator --------------------- #
# ---------------------------------------------------------------------------- #

def _build_batch(ds, idxs):
    return [ds[i] for i in idxs]


def test_padding_and_attention_mask(json_dataset, tokenizer, vision_tower):
    from friday.data import PretrainingDataset, FridayCollator
    json_path, img_dir, _ = json_dataset
    ds = PretrainingDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    batch = _build_batch(ds, [0, 1])

    collator = FridayCollator(tokenizer=tokenizer)
    out = collator(batch)

    assert out["input_ids"].shape == out["labels"].shape
    # attention_mask true == non‑pad positions
    mask = out["attention_mask"]
    assert torch.all(mask == out["input_ids"].ne(tokenizer.pad_token_id))


def test_eos_pad_roundtrip(json_dataset, tokenizer, vision_tower):
    from friday.data import PretrainingDataset, FridayCollator
    # make tokenizer where pad==eos
    collator = FridayCollator(tokenizer=tokenizer)

    json_path, img_dir, _ = json_dataset
    ds = PretrainingDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    batch = _build_batch(ds, [0])

    out = collator(batch)
    # after in‑place restoration, PAD_FOR_EOS should no longer be present
    assert not torch.any(out["input_ids"] == -123)    # PAD_FOR_EOS constant


def test_image_list_batch_size(json_dataset, tokenizer, vision_tower):
    from friday.data import PretrainingDataset, FridayCollator
    json_path, img_dir, _ = json_dataset
    ds = PretrainingDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    batch = _build_batch(ds, [0, 1, 0])
    out = FridayCollator(tokenizer=tokenizer)(batch)

    assert len(out["images"]) == 3
    # inner list lengths preserved (=1 image each sample)
    assert all(len(lst) == 1 for lst in out["images"])
