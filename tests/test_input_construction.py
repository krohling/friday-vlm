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

# ---------------------------------------------------------------------------------- #
# ----------------------------  Dummy helper classes  ------------------------------ #
# ---------------------------------------------------------------------------------- #
class DummyProjector(torch.nn.Module):
    """Stand‑in for MLPAdapter; returns all‑zero projections."""
    def __init__(self, output_dim: int = 4):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        shape = (*x.shape[:-1], self.output_dim)
        return torch.zeros(shape, dtype=x.dtype, device=x.device)


class DummyVisionTower:
    """Stand‑in for a Siglip tower with a trivial `preprocess_images`."""
    def __init__(self):
        # minimal attributes used by Friday code
        self.device = torch.device("cpu")

    def preprocess_images(self, imgs: List[Image.Image], pad_and_stack_tensors=True):
        # produce a (N,3,32,32) tensor filled with zeros
        return torch.zeros(len(imgs), 3, 32, 32)

# ---------------------------------------------------------------------------------- #
# ----------------------------  Lightweight model patching ------------------------- #
# ---------------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_friday(monkeypatch):
    """
    Before importing `FridayForCausalLM`, patch away heavyweight components so
    its helper methods can be unit‑tested with tiny tensors.
    """
    # --- Patch the vision‑related classes ---------------------------------------- #
    import friday.model.vision_tower as vt
    import friday.model.vision_adapter as va

    monkeypatch.setattr(vt, "SiglipVisionTower", DummyVisionTower, raising=True)
    monkeypatch.setattr(vt, "SiglipVisionTowerS2", DummyVisionTower, raising=True)
    monkeypatch.setattr(va, "MLPAdapter", DummyProjector, raising=True)

    # --- Patch FridayForCausalLM.__init__ with a lightweight version ------------- #
    from friday.model import FridayForCausalLM, FridayConfig

    def _light_init(self, config: FridayConfig):
        torch.nn.Module.__init__(self)
        # store config / device
        self.config = config
        self.to(torch.device("cpu"))

        hidden = 8
        vocab  = 1000
        config.hidden_size = hidden

        # tiny embedding layer used by get_multimodal_input_embeddings
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)

        # build a “mini‑inner model” carrying only the attributes used by helpers
        class _Inner(torch.nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.embed_tokens = outer.embed_tokens
                self.mm_projector = DummyProjector(output_dim=4)
                self.vision_tower = DummyVisionTower()

            # very small image‑feature generator: (N, 1, 4) zeros
            def compute_image_features(self, imgs):
                n_imgs = imgs.shape[0] if torch.is_tensor(imgs) else len(imgs)
                return torch.zeros(n_imgs, 1, 4)

            # required by FridayForCausalLM.get_vision_tower()
            def get_vision_tower(self):
                return self.vision_tower

            def parameters(self, recurse=True):
                return []

        self.model = _Inner(self)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

        # copy special‑token IDs onto the outer object (helpers rely on these)
        self.image_token_id  = config.cfg_special_tokens["image_token_id"]
        self.image_start_id  = config.cfg_special_tokens["image_start_token_id"]
        self.image_end_id    = config.cfg_special_tokens["image_end_token_id"]

    monkeypatch.setattr(FridayForCausalLM, "__init__", _light_init, raising=True)


# reusable helper: build a lightweight Friday object
@pytest.fixture
def friday():
    from friday.model import FridayForCausalLM, FridayConfig

    cfg = FridayConfig(delay_load=True)
    return FridayForCausalLM(cfg)


# ---------------------------------------------------------------------------------- #
# -----------------------  5.1 get_multimodal_input_embeddings ---------------------- #
# ---------------------------------------------------------------------------------- #
def test_embedding_injection_single_img(friday):
    IGNORE_INDEX = -100
    image_tok = friday.image_token_id
    ids = torch.tensor([10, image_tok, 11])

    # one image → shape (1, 1, 4)
    image_feats = [torch.zeros(1, 1, 4)]

    embeds, labels = friday.get_multimodal_input_embeddings(
        [ids], image_feats, return_labels=True
    )

    # The single sequence should expand: 3 original tokens → 5 embeddings
    #   10  <img_start> [patch] <img_end> 11
    assert embeds[0].shape[0] == 5

    # Label positions 1‑3 (img_start, patch, img_end) must be masked out
    masked = labels[0][1:4]
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
    batch_ids = [
        torch.tensor([1, img_tok, 2]),                        # 1 image
        torch.tensor([3, img_tok, 4, img_tok, 5])            # 2 images
    ]
    image_feats = [
        torch.zeros(1, 1, 4),                                # for first row
        torch.zeros(2, 1, 4),                                # for second row
    ]
    embeds, labels = friday.get_multimodal_input_embeddings(
        batch_ids, image_feats, return_labels=True
    )

    # correct per‑row lengths
    assert embeds[0].shape[0] == 5           # 1-> 1+2 extra
    assert embeds[1].shape[0] == 8           # 2-> 2*2 extra


# ---------------------------------------------------------------------------------- #
# --------  5.2 prepare_inputs_for_multimodal : various image input types ----------- #
# ---------------------------------------------------------------------------------- #
# helper to create a dummy PIL image
def _dummy_pil():
    return Image.new("RGB", (32, 32), color=(0, 0, 0))


@pytest.mark.parametrize(
    "img_arg,batch",
    [
        ([[ _dummy_pil(), _dummy_pil() ]],                     1),               # list[list[PIL]]
        ([_dummy_pil(), _dummy_pil()],                         1),               # list[PIL]
        ([torch.zeros(2, 3, 32, 32)],                          1),               # list[tensor]
        (_dummy_pil(),                                          1),               # single PIL
    ]
)
def test_prepare_inputs_various_types(friday, img_arg, batch):
    img_tok = friday.image_token_id
    input_ids = torch.tensor([[img_tok]] * batch)
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
