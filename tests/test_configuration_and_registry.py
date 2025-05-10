# test_configuration_and_registry.py
#
# Unit‑tests for the “Configuration & Registry” subsystem.
# Run with:  pytest -q test_configuration_and_registry.py
#
# These tests are deliberately lightweight: they do **not** download model
# checkpoints and do not allocate large tensors.  All heavyweight initialisation
# in `FridayForCausalLM` is monkey‑patched away.

import copy
import pytest
from transformers import AutoConfig, AutoModelForCausalLM

# --------------------------------------------------------------------------- #
# Robust import of the Friday symbols (package layout may differ per project)
# --------------------------------------------------------------------------- #
from friday.model import (
    FridayConfig,
    FridayForCausalLM,
)

from friday.model.friday_arch import (
    DEFAULT_CFG_VISION_TOWER,
    DEFAULT_CFG_VISION_ADAPTER,
    DEFAULT_CFG_SPECIAL_TOKENS
)


# --------------------------------------------------------------------------- #
# 1. FridayConfig default behaviour
# --------------------------------------------------------------------------- #
def test_friday_config_defaults():
    """Creating FridayConfig() without overrides must copy‑populate defaults."""
    cfg = FridayConfig()                        # delay_load=True by default

    # Dicts equal in *value* but are *different objects* (defensive copy).
    assert cfg.cfg_vision_tower == DEFAULT_CFG_VISION_TOWER
    assert cfg.cfg_vision_tower is not DEFAULT_CFG_VISION_TOWER

    assert cfg.cfg_vision_adapter == DEFAULT_CFG_VISION_ADAPTER
    assert cfg.cfg_vision_adapter is not DEFAULT_CFG_VISION_ADAPTER

    assert cfg.cfg_special_tokens == DEFAULT_CFG_SPECIAL_TOKENS
    assert cfg.cfg_special_tokens is not DEFAULT_CFG_SPECIAL_TOKENS


# --------------------------------------------------------------------------- #
# 2. FridayConfig override & merge behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "vision_override, special_override",
    [
        ({"use_s2": False}, {"image_token_id": 42}),
        ({"pad_to_square": False, "model_name_or_path": "dummy/path"},
         {"image_start_token_id": 123456}),
    ],
)
def test_friday_config_override(vision_override, special_override):
    """Partial override dictionaries must merge with defaults, not replace."""
    cfg = FridayConfig(
        cfg_vision_tower=vision_override,
        cfg_special_tokens=special_override,
    )

    # Vision tower: overridden keys updated, others stay default
    for k, v in vision_override.items():
        assert cfg.cfg_vision_tower[k] == v
    for k, v in DEFAULT_CFG_VISION_TOWER.items():
        if k not in vision_override:
            assert cfg.cfg_vision_tower[k] == v

    # Special tokens: same logic
    for k, v in special_override.items():
        assert cfg.cfg_special_tokens[k] == v
    for k, v in DEFAULT_CFG_SPECIAL_TOKENS.items():
        if k not in special_override:
            assert cfg.cfg_special_tokens[k] == v



# --------------------------------------------------------------------------- #
# 3. Test instantiation behaviour
# --------------------------------------------------------------------------- #
def test_instantiate_with_config():
    """Instantiate the model and confirm that frozen parameters have requires_grad disabled."""
    cfg = FridayConfig()
    model = FridayForCausalLM(cfg)
    model.initialize_vision_modules()

    assert isinstance(model, FridayForCausalLM)
    assert isinstance(model.config, FridayConfig)

def test_instantiate_from_pretrained():
    """Instantiate the model and confirm that frozen parameters have requires_grad disabled."""
    model = FridayForCausalLM.from_pretrained("microsoft/Phi-4-mini-reasoning")
    model.initialize_vision_modules()

    assert isinstance(model, FridayForCausalLM)
    assert isinstance(model.config, FridayConfig)

def test_instantiate_with_auto_config():
    """Instantiate the model and confirm that frozen parameters have requires_grad disabled."""
    cfg = AutoConfig.for_model("friday-phi")
    model = AutoModelForCausalLM.from_config(cfg)
    model.initialize_vision_modules()
    
    assert isinstance(model, FridayForCausalLM)
    assert isinstance(model.config, FridayConfig)



# --------------------------------------------------------------------------- #
# 4. Test freeze behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "freeze_llm, freeze_vision_tower, freeze_vision_adapter",
    [
        (True, True, True),  # all freeze
        (False, False, False),  # all unfreeze
        (True, False, False),  # LLM freeze, others unfreeze
        (False, True, True),  # LLM unfreeze, others freeze
    ],
)
def test_friday_config_freeze(
    freeze_llm, freeze_vision_tower, freeze_vision_adapter
):
    """Instantiate the model and confirm that frozen parameters have requires_grad disabled."""
    cfg = FridayConfig(
        freeze_llm=freeze_llm,
        cfg_vision_tower={"freeze": freeze_vision_tower},
        cfg_vision_adapter={"freeze": freeze_vision_adapter},        
    )

    model = FridayForCausalLM(cfg)
    model.initialize_vision_modules()

    for name, param in model.named_parameters():
        if "vision_tower" in name:
            assert param.requires_grad == (not freeze_vision_tower)
        elif "mm_projector" in name:
            assert param.requires_grad == (not freeze_vision_adapter)
        else:
            assert param.requires_grad == (not freeze_llm)
    
    



# --------------------------------------------------------------------------- #
# 5. HuggingFace Auto* class registry integration
# --------------------------------------------------------------------------- #
def test_auto_registry(monkeypatch):
    """Ensure the config & model classes are discoverable through the HF Auto API.

    Heavy construction in `FridayForCausalLM.__init__` is patched out so the
    test runs instantly and without GPU/weights.
    """

    # --- Patch FridayForCausalLM.__init__ to be a no‑op -------------------- #
    def _lightweight_init(self, config):
        self.config = config  # minimal state; skip super‑calls / weights

    # monkeypatch.patch.object(FridayForCausalLM, "__init__", _lightweight_init)
    monkeypatch.setattr(FridayForCausalLM, "__init__", _lightweight_init, raising=True)

    # --- AutoConfig should instantiate the correct config class ------------ #
    cfg = AutoConfig.for_model("friday-phi")
    assert isinstance(cfg, FridayConfig)

    # --- AutoModelForCausalLM should return the Friday model class --------- #
    model = AutoModelForCausalLM.from_config(cfg)
    assert isinstance(model, FridayForCausalLM)
    # and model carries back the same config instance

    compare_keys = [k for k in cfg.to_dict().keys() if not k.startswith("_")]
    for k in compare_keys:
        assert cfg.to_dict()[k] == model.config.to_dict()[k], f"Key {k} mismatch"
