import os
import tempfile
import torch
import pytest
from easydict import EasyDict

from peft.tuners.lora import LoraLayer
from friday.train.config import FridayTrainingArguments
from friday.train.model_factory import build_model      # adjust import path if needed

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

TOKENIZER_CFG = {"pretrained_model_name_or_path": "dummy"}

def targs(**kw):
    base = dict(
        bits                = 16,
        fp16                = True,
        bf16                = False,
        lora_enable         = False,
        lora_params         = {"r":4, "lora_alpha":8, "lora_dropout":0.0, "bias":"none"},
        gradient_checkpointing = False,
        freeze_language_model   = False,
        freeze_vision_tower     = False,
        freeze_vision_adapter   = False,
        bits_and_bytes_params   = {},
    )
    base.update(kw)
    return FridayTrainingArguments(**base)

# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


def test_lora_injection_fp16():
    m_cfg = {}
    args  = targs(lora_enable=True)

    model, _ = build_model(m_cfg, TOKENIZER_CFG, args)

    adapters = [m for m in model.modules() if isinstance(m, LoraLayer)]
    assert adapters, "LoRA was not injected"
    assert all(p.dtype == torch.float16
               for n, p in model.named_parameters() if 'lora_' in n)


def test_norm_cast_fp32():
    m_cfg = {}
    args  = targs()                       # default 16-bit

    model, _ = build_model(m_cfg, TOKENIZER_CFG, args)

    assert all(p.dtype == torch.float32
               for n, p in model.named_parameters() if 'norm' in n)


def test_kbit_guard_raises_on_fp32():
    m_cfg = {}
    args  = targs(bits=4, fp16=False, bf16=False)

    with pytest.raises(ValueError):
        build_model(m_cfg, TOKENIZER_CFG, args)


def test_freeze_flags():
    m_cfg = {}
    args  = targs(freeze_language_model=True,
                  freeze_vision_tower=True,
                  freeze_vision_adapter=True)

    model, _ = build_model(m_cfg, TOKENIZER_CFG, args)

    assert all(not p.requires_grad for p in model.parameters())


def test_mm_projector_checkpoint_updates_cfg(tmp_path):
    # create fake checkpoint file
    ckpt = tmp_path / "mm_projector.bin"
    ckpt.touch()

    m_cfg = {}
    args  = targs()

    _model, _ = build_model(m_cfg, TOKENIZER_CFG, args,
                            mm_projector_checkpoint=str(ckpt))

    assert m_cfg["vision_adapter"]["checkpoint_path"] == str(ckpt)


def test_mm_projector_missing_raises(tmp_path):
    missing = tmp_path / "nope.bin"
    m_cfg = {}
    args  = targs()

    with pytest.raises(ValueError):
        build_model(m_cfg, TOKENIZER_CFG, args,
                    mm_projector_checkpoint=str(missing))
