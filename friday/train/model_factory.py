import os

import torch

import transformers

from peft.tuners.lora import LoraLayer
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training
)

from friday.model import *
from friday.train.config import FridayTrainingArguments


def build_model(
        model_config: dict,
        tokenizer_config: dict,
        training_args: FridayTrainingArguments,
        mm_projector_checkpoint: str | None = None,
) -> tuple[FridayForCausalLM, transformers.PreTrainedTokenizerBase]:
    # ------ 1. Configure quantization and dtype ------
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args = dict(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                compute_dtype=compute_dtype,
                **training_args.bits_and_bytes_params
            )
        )
    



    # ------ 2. Load model, tokenizer, and vision tower ------
    if mm_projector_checkpoint is not None:
        if os.path.exists(mm_projector_checkpoint):
            if "vision_adapter" in model_config:
                model_config["vision_adapter"]["checkpoint_path"] = mm_projector_checkpoint
            else:
                model_config["vision_adapter"] = dict(checkpoint_path=mm_projector_checkpoint)
        else:
            raise ValueError(f"MM Projector checkpoint {mm_projector_checkpoint} does not exist.")


    tokenizer = transformers.AutoTokenizer.from_pretrained(**tokenizer_config)
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = FridayForCausalLM.from_pretrained(
        **model_config,
        torch_dtype=compute_dtype,
        tokenizer_model_max_length=tokenizer.model_max_length,
        **bnb_model_from_pretrained_args,
    )
    model.initialize_vision_modules()
    model.set_llm_dtype(compute_dtype)
    model.set_vision_tower_dtype(compute_dtype)
    model.set_vision_adapter_dtype(compute_dtype)
    
    
    
    # ------ 3. Configure model for training ------
    if training_args.bits in [4, 8]:
        if compute_dtype == torch.float32:
            raise ValueError("4/8-bit quantisation requires fp16 or bf16 compute_dtype")
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    elif training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        target_modules = set()
        for name, module in model.get_llm_named_modules().items():
            if isinstance(module, torch.nn.Linear):
                names = name.split('.')
                parsed_name = names[0] if len(names) == 1 else names[-1]
                if 'lm_head' not in parsed_name:
                    target_modules.add(parsed_name)

        lora_config = LoraConfig(
            target_modules=sorted(target_modules),
            task_type="CAUSAL_LM",
            **training_args.lora_params
        )
        model = get_peft_model(model, lora_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(dtype=compute_dtype)
        elif 'norm' in name:
            module.to(torch.float32)
        elif 'lm_head' in name or 'embed_tokens' in name:
            if training_args.bf16 and hasattr(module, 'weight') and module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)
    

    model.set_llm_requires_grad(not training_args.freeze_language_model, exclude_lora=True)
    model.set_vision_tower_requires_grad(not training_args.freeze_vision_tower)
    model.set_vision_adapter_requires_grad(not training_args.freeze_vision_adapter)

    return model, tokenizer
