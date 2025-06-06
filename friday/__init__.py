# friday/__init__.py
from .model.friday_arch import FridayConfig, FridayForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register(FridayConfig.model_type, FridayConfig)
AutoModelForCausalLM.register(FridayConfig, FridayForCausalLM)

__all__ = ["FridayConfig", "FridayForCausalLM"]
