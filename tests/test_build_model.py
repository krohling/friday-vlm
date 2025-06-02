import os
import tempfile
import torch
import pytest
from easydict import EasyDict

from peft.tuners.lora import LoraLayer
from friday.train.config import FridayTrainingArguments
from friday.train.model_factory import build_model

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
    
    assert not any(isinstance(m, LoraLayer) for m in model.modules())

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

    lm_head_params = [(n,p) for n, p in model.named_parameters() if 'lm_head' in n]
    assert lm_head_params, "LM head parameters not found"
    assert all('lora_' not in n for n, p in lm_head_params), \
           "Expected LM head parameters to not be LoRA parameters, but found LoRA in names."
    
    vt_params = [(n,p) for n, p in model.named_parameters() if 'vision_tower' in n]
    assert vt_params, "Vision Tower parameters not found"
    assert all('lora_' not in n for n, p in vt_params), \
           "Expected Vision Tower parameters to not be LoRA parameters, but found LoRA in names."
    
    va_params = [(n,p) for n, p in model.named_parameters() if 'mm_projector' in n]
    assert va_params, "Vision Adapter parameters not found"
    assert all('lora_' not in n for n, p in va_params), \
           "Expected Vision Adapter parameters to not be LoRA parameters, but found LoRA in names."
    




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



@pytest.mark.parametrize(
    "t_args, p_name, lora_dtype",
    [
        ({
            "bits": 4,
            "bf16": True,
            "lora_enable": True,
            "lora_params": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"},
            "bits_and_bytes_params": {"strict_load": False},
            "freeze_language_model": True,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": True,
        }, "4bit", torch.bfloat16),
        ({
            "bits": 4,
            "fp16": True,
            "lora_enable": True,
            "lora_params": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"},
            "bits_and_bytes_params": {"strict_load": False},
            "freeze_language_model": True,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": True,
        }, "4bit", torch.float16),
        ({
            "bits": 8,
            "bf16": True,
            "lora_enable": True,
            "lora_params": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"},
            "bits_and_bytes_params": {"strict_load": False},
            "freeze_language_model": True,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": True,
        }, "int8", torch.bfloat16),
        ({
            "bits": 8,
            "fp16": True,
            "lora_enable": True,
            "lora_params": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"},
            "bits_and_bytes_params": {"strict_load": False},
            "freeze_language_model": True,
            "freeze_vision_tower": True,
            "freeze_vision_adapter": True,
        }, "int8", torch.float16),
    ],
)
def test_quantization_and_lora_configuration(t_args, p_name, lora_dtype):
    model, _ = build_model(
        MODEL_CFG, 
        TOKENIZER_CFG, 
        FridayTrainingArguments(**t_args)
    )

    # print_params = [
    #     (n,p) for n, p in model.named_parameters() 
    #     if "vision_tower" not in n and "mm_projector" not in n and "lora_" not in n and "norm" not in n and "lm_head" not in n
    # ]
    # for n, p in print_params:
    #     print(f"{n}: {type(p).__name__}, dtype: {p.dtype}")

    assert all(
            p_name in type(p).__name__.lower()
            for n,p in model.named_parameters() if "vision_tower" not in n and "mm_projector" not in n and "lora_" not in n and "norm" not in n and "lm_head" not in n
        ), "Expected all LLM parameters (except norm and lm_head layers) to be quantized, but found non-quantized types."

    assert all(
            p.requires_grad
            for n, p in model.named_parameters() if "lora_" in n
        ), "Expected all LoRA parameters to have requires_grad=True, but found some with requires_grad=False."
    
    assert all(
            not p.requires_grad
            for n, p in model.named_parameters() if "lora_" not in n
        ), "Expected all non-LoRA parameters to have requires_grad=False, but found some with requires_grad=True."
    
    assert all(
            p_name not in type(p).__name__.lower()
            for n,p in model.named_parameters() if "vision_tower" in n
        ), "Expected all Vision Tower parameters not to be quantized, but found quantized types."

    assert all(
            p_name not in type(p).__name__.lower()
            for n,p in model.named_parameters() if "mm_projector" in n
        ), "Expected all Vision Adapter parameters not to be quantized, but found quantized types."
    
    assert all(
            p_name not in type(p).__name__.lower()
            for n,p in model.named_parameters() if "norm" in n
        ), "Expected all norm parameters not to be quantized, but found quantized types."
    
    assert all(
            p_name not in type(p).__name__.lower()
            for n,p in model.named_parameters() if "lm_head" in n
        ), "Expected all lm_head parameters not to be quantized, but found quantized types."
    
    assert all(
        p.dtype is lora_dtype
        for n, p in model.named_parameters() if "lora_" in n
    )



