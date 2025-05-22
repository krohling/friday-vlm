# test_data_pipeline.py
#
# Unit tests for the data‑pipeline utilities:
#   • preprocess_for_finetuning
#   • FinetuningDataset
#   • FinetuningCollator
#
# Heavy external dependencies (real tokenizer, SigLIP tower, etc.) are replaced
# with fast stubs so the whole file executes in < 1 s on CPU.
#
# Run with:  pytest -q test_data_pipeline.py
#
import json
from pathlib import Path

import pytest
import torch

from util import test_images_paths, build_test_images_paths

from transformers import AutoTokenizer
from friday.constants import IGNORE_INDEX


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
# -----------------------  Helpers for temporary sample data ----------------- #
# ---------------------------------------------------------------------------- #

EX_SYSTEM_MESSAGE = "You are a helpful AI assistant."

def _make_sample(img_file=None):
    return {
        "image": str(img_file) if img_file else None,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nPlease provide a short description for this region: [0.47, 0.15, 0.52, 0.19]." if img_file else "Please provide a short description for this region: [0.47, 0.15, 0.52, 0.19]."
            },
            {
                "from": "gpt",
                "value": "Basket is on the bike."
            },
            {
                "from": "human",
                "value": "Please provide the bounding box coordinate of the region this sentence describes: food in the container."
            },
            {
                "from": "gpt",
                "value": "[0.28, 0.8, 0.48, 0.98]"
            },
            {
                "from": "human",
                "value": "Please provide a short description for this region: [0.4, 0.09, 0.57, 0.17]."
            },
            {
                "from": "gpt",
                "value": "Brown wicker basket on bike."
            },
            {
                "from": "human",
                "value": "Please provide the bounding box coordinate of the region this sentence describes: Male wearing blue shirt."
            },
            {
                "from": "gpt",
                "value": "[0.64, 0.41, 0.86, 1.0]"
            },
            {
                "from": "human",
                "value": "Please provide the bounding box coordinate of the region this sentence describes: Assorted type of food by person."
            },
            {
                "from": "gpt",
                "value": "[0.5, 0.78, 0.69, 0.95]"
            },
            {
                "from": "human",
                "value": "Please provide the bounding box coordinate of the region this sentence describes: white container filled with food."
            },
            {
                "from": "gpt",
                "value": "[0.15, 0.8, 0.5, 0.97]"
            },
            {
                "from": "human",
                "value": "Please provide a short description for this region: [0.76, 0.25, 0.79, 0.28]."
            },
            {
                "from": "gpt",
                "value": "Patch of green grass."
            },
            {
                "from": "human",
                "value": "Please provide a short description for this region: [0.61, 0.87, 0.7, 0.92]."
            },
            {
                "from": "gpt",
                "value": "Round food in plate."
            },
            {
                "from": "human",
                "value": "Please provide a short description for this region: [0.45, 0.11, 0.55, 0.18]."
            },
            {
                "from": "gpt",
                "value": "Basket on the bike."
            },
            {
                "from": "human",
                "value": "Please provide a short description for this region: [0.42, 0.75, 0.49, 0.85]."
            },
            {
                "from": "gpt",
                "value": "Orange soda by person."
            }
            ],
        }

# ---------------------------------------------------------------------------- #
# ---------------------------  7.1 preprocess_for_finetuning ---------------- #
# ---------------------------------------------------------------------------- #

# parameterize
@pytest.mark.parametrize("sample", [
    _make_sample(build_test_images_paths()[0]),
    _make_sample(),
])
def test_prompt_construction(tokenizer, vision_tower, test_images_paths, sample):
    from friday.train.data.finetuning import preprocess_for_finetuning
    out = preprocess_for_finetuning(sample, str(test_images_paths[0].parent), vision_tower, tokenizer, EX_SYSTEM_MESSAGE)

    system_tok_id = tokenizer.vocab["<|system|>"]
    sys_token_count = (out["input_ids"] == system_tok_id).sum().item()
    assert sys_token_count == 1

    img_tok_id = tokenizer.vocab["<image>"]
    image_tag_count = sum(1 for conv in sample["conversations"] if 'image' in conv['value'])
    image_token_count = (out["input_ids"] == img_tok_id).sum().item()
    assert image_token_count == image_tag_count

    user_tok_id = tokenizer.vocab["<|user|>"]
    user_message_count = sum(1 for conv in sample["conversations"] if conv['from'] == 'human')
    user_token_count = (out["input_ids"] == user_tok_id).sum().item()
    assert user_token_count == user_message_count

    asst_tok_id = tokenizer.vocab["<|assistant|>"]
    asst_message_count = sum(1 for conv in sample["conversations"] if conv['from'] == 'gpt')
    asst_token_count = (out["input_ids"] == asst_tok_id).sum().item()
    assert asst_token_count == asst_message_count

    end_tok_id = tokenizer.vocab["<|end|>"]
    end_tok_count = (out["input_ids"] == end_tok_id).sum().item()
    assert end_tok_count == len(sample["conversations"]) + 1



@pytest.mark.parametrize("sample", [
    _make_sample(build_test_images_paths()[0]),
    _make_sample(),
])
def test_label_masking(test_images_paths, tokenizer, vision_tower, sample):
    from friday.train.data.finetuning import preprocess_for_finetuning
    out = preprocess_for_finetuning(sample, str(test_images_paths[0].parent), vision_tower, tokenizer, EX_SYSTEM_MESSAGE)

    inside = False
    system_tok_id = tokenizer.vocab["<|system|>"]
    img_tok_id = tokenizer.vocab["<image>"]
    user_tok_id = tokenizer.vocab["<|user|>"]
    asst_tok_id = tokenizer.vocab["<|assistant|>"]
    end_tok_id = tokenizer.vocab["<|end|>"]

    all_special_tokens = [
        system_tok_id, img_tok_id, user_tok_id, asst_tok_id, end_tok_id
    ]

    for i, t in enumerate(out['input_ids'][:-1]):
        if t in all_special_tokens:
            assert out['labels'][i] == IGNORE_INDEX
            
            if t in [system_tok_id, user_tok_id]:
                inside = True
            elif t == end_tok_id:
                inside = False
        elif inside:
            assert out['labels'][i] == IGNORE_INDEX
        else:
            assert out['labels'][i] != IGNORE_INDEX
    
    # check that the last token is an end token
    assert out['labels'][-1] == end_tok_id



@pytest.mark.parametrize("sample", [
    _make_sample(build_test_images_paths()[0]),
    _make_sample(),
])
def test_image_preprocessing(tokenizer, vision_tower, test_images_paths, sample):
    from friday.train.data.finetuning import preprocess_for_finetuning
    out = preprocess_for_finetuning(sample, str(test_images_paths[0].parent), vision_tower, tokenizer, EX_SYSTEM_MESSAGE)

    # check that the image is a list of tensors
    assert isinstance(out["image"], list)
    assert all(isinstance(img, torch.Tensor) for img in out["image"])
    if sample['image'] is None:
        assert len(out["image"]) == 0

    # check that the image tensor has the expected shape
    output_size = vision_tower.image_processor.size
    for img in out["image"]:
        assert img.shape == (3, output_size["height"], output_size["width"])


def test_missing_image(tokenizer, vision_tower, test_images_paths):
    from friday.train.data.finetuning import preprocess_for_finetuning
    sample = _make_sample(test_images_paths[0])
    sample["conversations"][2]["value"] = sample["conversations"][2]["value"] + "<image>"

    with pytest.raises(ValueError):
        preprocess_for_finetuning(sample, str(test_images_paths[0].parent), vision_tower, tokenizer, EX_SYSTEM_MESSAGE)


# ---------------------------------------------------------------------------- #
# -----------------------------  7.2 FinetuningDataset ---------------------- #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def json_dataset(tmp_path: Path, test_images_paths):
    """Write a JSON file with two samples and return its path."""
    img0, img1 = test_images_paths
    samples = [_make_sample(img0), _make_sample()]
    json_path = tmp_path / "samples.json"
    json_path.write_text(json.dumps(samples))
    return json_path, img0.parent, len(samples)


def test_len_matches_json(json_dataset, tokenizer, vision_tower):
    from friday.train.data.finetuning import FinetuningDataset

    json_path, img_dir, N = json_dataset
    ds = FinetuningDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    assert len(ds) == N


def test_get_item_keys_types(json_dataset, tokenizer, vision_tower):
    from friday.train.data.finetuning import FinetuningDataset

    json_path, img_dir, _ = json_dataset
    ds = FinetuningDataset(
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
# -----------------------------  7.3 FinetuningCollator --------------------- #
# ---------------------------------------------------------------------------- #

def _build_batch(ds, idxs):
    return [ds[i] for i in idxs]


def test_padding_and_attention_mask(json_dataset, tokenizer, vision_tower):
    from friday.train.data.finetuning import FinetuningDataset, FinetuningCollator
    json_path, img_dir, _ = json_dataset
    ds = FinetuningDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    batch = _build_batch(ds, [0, 1])

    collator = FinetuningCollator(tokenizer=tokenizer)
    out = collator(batch)

    assert out["input_ids"].shape == out["labels"].shape
    # attention_mask true == non‑pad positions
    mask = out["attention_mask"]
    assert torch.all(mask == out["input_ids"].ne(tokenizer.pad_token_id))


def test_eos_pad_roundtrip(json_dataset, tokenizer, vision_tower):
    from friday.train.data.finetuning import FinetuningDataset, FinetuningCollator
    # make tokenizer where pad==eos
    # tok = DummyTokenizer(pad_id=2, eos_id=2)
    collator = FinetuningCollator(tokenizer=tokenizer)

    json_path, img_dir, _ = json_dataset
    ds = FinetuningDataset(
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
    from friday.train.data.finetuning import FinetuningDataset, FinetuningCollator
    json_path, img_dir, _ = json_dataset
    ds = FinetuningDataset(
        data_path=str(json_path),
        image_dir=str(img_dir),
        tokenizer=tokenizer,
        vision_tower=vision_tower,
    )
    batch = _build_batch(ds, [0, 1, 0])
    out = FinetuningCollator(tokenizer=tokenizer)(batch)

    assert len(out["images"]) == 3
    assert len(out["images"][0]) == 1
    assert len(out["images"][1]) == 0
    assert len(out["images"][2]) == 1
