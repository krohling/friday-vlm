# test_multimodal_input.py
#
# Unit‑tests for the “Multimodal Embedding & Input Construction” logic inside
# friday.model.FridayForCausalLM.
#
#  * Heavy language‑model, vision‑tower and adapter classes are replaced with
#    tiny stubs so the whole file executes in <1 s on CPU.
#  * Only the two helper methods `get_multimodal_input_embeddings` and
#    `prepare_inputs_for_multimodal` are exercised.
#
# Run with:  pytest -q test_multimodal_input.py
#
import types
from typing import List

import pytest
import torch
from PIL import Image


# reusable helper: build a lightweight Friday object
@pytest.fixture
def friday():
    from friday.model import FridayForCausalLM, FridayConfig

    cfg = FridayConfig(delay_load=True)
    model = FridayForCausalLM(cfg)
    model.initialize_vision_modules()
    return model


# ---------------------------------------------------------------------------------- #
# -----------------------  5.1 get_multimodal_input_embeddings ---------------------- #
# ---------------------------------------------------------------------------------- #
def test_embedding_injection_single_img(friday):
    IGNORE_INDEX = -100
    image_tok = friday.image_token_id
    ids = torch.tensor([10, image_tok, 11])

    embedding_dim = friday.config.cfg_vision_adapter["output_dim"]
    image_feats = [torch.zeros(1, 1, embedding_dim)]

    embeds, labels = friday.get_multimodal_input_embeddings(
        [ids], image_feats, return_labels=True
    )

    #   10  <img_start> [patch] <img_end> 11
    assert embeds[0].shape[0] == 5

    # Label positions 1‑3 (img_start, patch, img_end) must be masked out
    masked = labels[0][1:4]
    #   1: <img_start> 2: [patch] 3: <img_end>
    assert torch.all(masked == IGNORE_INDEX)
    # Others keep original ids
    assert labels[0][0].item() == 10 and labels[0][-1].item() == 11


def test_mismatch_token_count_raises(friday):
    ids = torch.tensor([friday.image_token_id, 11])
    # zero images supplied → mismatch
    with pytest.raises(ValueError):
        friday.get_multimodal_input_embeddings([ids], [torch.zeros(0, 1, 4)])


def test_multiple_images_in_batch(friday):
    img_tok = friday.image_token_id
    embedding_dim = friday.config.cfg_vision_adapter["output_dim"]
    batch_ids = [
        torch.tensor([1, img_tok, 2]),                        # 1 image
        torch.tensor([3, img_tok, 4, img_tok, 5])            # 2 images
    ]
    image_feats = [
        torch.zeros(1, 1, embedding_dim),                                # for first row
        torch.zeros(2, 1, embedding_dim),                                # for second row
    ]
    embeds, labels = friday.get_multimodal_input_embeddings(
        batch_ids, image_feats, return_labels=True
    )

    # correct per‑row lengths
    assert embeds[0].shape[0] == 2 + (1 * 3) # 2 tokens + 1 image
    assert embeds[1].shape[0] == 3 + (2 * 3) # 3 tokens + 2 images

    # correct label values
    assert labels[0][0].item() == 1 and labels[0][-1].item() == 2
    assert labels[1][0].item() == 3 and labels[1][4].item() == 4 and labels[1][-1].item() == 5

    # Label positions 1‑3 (img_start, patch, img_end) must be masked out
    masked = labels[0][1:4]
    assert torch.all(masked == -100)
    masked = labels[1][1:4]
    assert torch.all(masked == -100)
    masked = labels[1][5:8]
    assert torch.all(masked == -100)
    


# ---------------------------------------------------------------------------------- #
# --------  5.2 prepare_inputs_for_multimodal : various image input types ----------- #
# ---------------------------------------------------------------------------------- #
# helper to create a dummy PIL image
def _dummy_pil():
    return Image.new("RGB", (32, 32), color=(0, 0, 0))


@pytest.mark.parametrize(
    "img_arg,img_token_count,batch",
    [
        ([[ _dummy_pil(), _dummy_pil() ]], 2,                   1),               # list[list[PIL]]
        ([_dummy_pil(), _dummy_pil()], 2,                       1),               # list[PIL]
        ([torch.zeros(2, 3, 32, 32)], 2,                        1),               # list[tensor]
        (_dummy_pil(), 1,                                       1),               # single PIL
    ]
)
def test_prepare_inputs_various_types(friday, img_arg, img_token_count, batch):
    img_tok = friday.image_token_id
    input_ids = torch.tensor([[img_tok]* img_token_count] * batch)  # (batch, seq_len)
    out = friday.prepare_inputs_for_multimodal(
        input_ids=input_ids,
        images=img_arg,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
    )

    (_, position_ids, attention_mask,
     _pkv, new_embeds, _labels) = out

    # Resulting tensors must match batch size
    assert attention_mask.shape[0] == batch
    assert new_embeds.shape[0] == batch
    # attention_mask true count equals sequence length per row
    lengths = attention_mask.sum(dim=1).tolist()
    for L in lengths:
        assert L == new_embeds.shape[1]


def test_attention_mask_correctness(friday):
    img_tok = friday.image_token_id
    ids = torch.tensor([[img_tok, 7, 8]])
    attn = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    (_, _pos, new_mask, _pkv, new_embeds, _labels) = friday.prepare_inputs_for_multimodal(
        input_ids=ids,
        images=[_dummy_pil()],
        position_ids=None,
        attention_mask=attn,
        past_key_values=None,
        labels=None,
    )

    # mask should have grown by +2 (img_start, img_end, patch = +3, -1 token removed)
    assert new_mask.shape[1] == new_embeds.shape[1]
    assert new_mask.sum().item() == new_embeds.shape[1]


def test_truncation_to_model_max_length(friday):
    # set very small limit
    friday.config.tokenizer_model_max_length = 6

    img_tok = friday.image_token_id
    ids = torch.tensor([[img_tok] * 2 + [9]])      # triggers >6 after expansion

    out = friday.prepare_inputs_for_multimodal(
        input_ids=ids,
        images=[[_dummy_pil(), _dummy_pil()]],      # two images
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
    )
    # new_embeds is element 4
    new_embeds = out[4]
    assert new_embeds.shape[1] <= 6
