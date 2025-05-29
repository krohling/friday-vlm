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


MODEL_CFG = {
    "pretrained_model_name_or_path": "microsoft/Phi-4-mini-reasoning",
    "cfg_vision_tower": {
        "pretrained_model_name_or_path": "kevin510/fast-vit-hd",
        "s2_scales": "512,1024",
        "type": "fastvit",
        "use_s2": True,
        "pad_to_square": True,
        "model_params": { "device_map": "cuda", "trust_remote_code": True }
    },
    "cfg_vision_adapter": {
        "input_dim": 1536,
        "hidden_dim": 512,
        "output_dim": 3072,
        "num_layers": 2,
        "activation": "gelu",
        "device": "cuda",
        "checkpoint_path": None
    }
}

TOKENIZER_CFG = {
    "pretrained_model_name_or_path": "kevin510/friday",
    "model_max_length": 2048,
    "padding_side": "right",
    "use_fast": True,
    "trust_remote_code": True
}

def targs(**kw):
    base = dict(
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
    return FridayTrainingArguments(**kw)

# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "t_args, dtype",
    [
        ({"fp16": True}, torch.float16),
        ({"bf16": True}, torch.bfloat16),
    ],
)
def test_dtype_configuration(t_args, dtype):
    model, _ = build_model(
        MODEL_CFG,
        TOKENIZER_CFG,
        FridayTrainingArguments(**t_args)
    )

    assert all(p.dtype == dtype for n, p in model.named_parameters() if 'norm' not in n), \
           f"Expected all parameters to be {dtype}, but found different dtypes."
    
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if 'norm' in n), \
           f"Expected all Norm parameters to be {torch.float32}, but found different dtypes."

@pytest.mark.parametrize(
    "t_args, dtype",
    [
        ({
            "fp16": True,
            "lora_enable": True,
            "lora_params": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"},
        }, torch.float16),
        ({
            "bf16": True,
            "lora_enable": True,
            "lora_params": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"},
        }, torch.bfloat16),
    ],
)
def test_lora_dtype_configuration(t_args, dtype):
    model, _ = build_model(
        MODEL_CFG,
        TOKENIZER_CFG,
        FridayTrainingArguments(**t_args)
    )

    adapters = [m for m in model.modules() if isinstance(m, LoraLayer)]
    assert adapters, "LoRA was not injected"
    assert all(
        p.dtype == dtype for m in adapters for n, p in m.named_parameters()
    ), "Expected all LoRA parameters to be {dtype}, but found different dtypes."



def test_kbit_guard_raises_on_fp32():
    with pytest.raises(ValueError):
        model, _ = build_model(
            MODEL_CFG,
            TOKENIZER_CFG,
            FridayTrainingArguments(**{
                "bits": 4,
                "fp16": False,
                "bf16": False,
            })
        )


@pytest.mark.parametrize(
    "t_args, lm_freeze, vt_freeze, va_freeze",
    [
        ({
            "freeze_language_model": True,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": True,
        }, True, True, True),
        ({
            "freeze_language_model": False,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": True,
        }, False, True, True),
        ({
            "freeze_language_model": True,
            "freeze_vision_tower": False,
            "freeze_vision_adapter": True,
        }, True, False, True),
        ({
            "freeze_language_model": True,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": False,
        }, True, True, False),
    ],
)
def test_freeze_flags(t_args, lm_freeze, vt_freeze, va_freeze):
    model, _ = build_model(
        MODEL_CFG,
        TOKENIZER_CFG,
        FridayTrainingArguments(**t_args)
    )

    assert all(p.requires_grad == (not lm_freeze) for p in model.get_llm_parameters())
    assert all(p.requires_grad == (not vt_freeze) for p in model.get_vision_tower().parameters())
    assert all(p.requires_grad == (not va_freeze) for p in model.get_vision_adapter().parameters())



def test_mm_projector_missing_raises(tmp_path):
    missing = tmp_path / "nope.bin"

    with pytest.raises(ValueError):
        build_model(MODEL_CFG, TOKENIZER_CFG, FridayTrainingArguments(),
                    mm_projector_checkpoint=str(missing))
