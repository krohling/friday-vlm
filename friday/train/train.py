import argparse
import os
import time

import json
from easydict import EasyDict

import torch

import transformers

from friday.model import *
from friday.data import PretrainingDataset, FridayCollator
from friday.train.friday_trainer import FridayTrainer, zip_and_upload_checkpoint_artifact
from friday.train.config import FridayTrainingArguments, FridayDataArguments
from friday.util import (
    find_all_linear_names, 
    get_peft_state_non_lora_maybe_zero_3, 
    get_peft_state_maybe_zero_3
)

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train():
    global local_rank

    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument('--config', type=str, help='The path to the config json file')
    parser.add_argument('--deepspeed', type=str, help='The path to the deepspeed config json file')
    parser.add_argument('--local_rank', type=int, help='The local rank for distributed training', default=-1)
    parser.add_argument('--resume_from_checkpoint', type=str, help='The path to the checkpoint to resume from', default=None)
    args = parser.parse_args()
    local_rank = args.local_rank

    # ------ 0. Load config ------
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    with open(args.config, 'r') as f:
        config = EasyDict(json.load(f))
    
    assert "tokenizer" in config, "Tokenizer config is required."
    assert "model" in config, "Model config is required."
    assert "data" in config, "Data config is required."
    assert "training" in config, "Training config is required."

    data_args = FridayDataArguments(**config.data)
    training_args = FridayTrainingArguments(**config.training)
    training_args.deepspeed = args.deepspeed if args.deepspeed else None

    if os.environ.get("BATCH_SIZE") is not None:
        training_args.per_device_train_batch_size = int(os.environ["BATCH_SIZE"])
    if os.environ.get("GRADIENT_ACCUMULATION_STEPS") is not None:
        training_args.gradient_accumulation_steps = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
    if os.environ.get("LEARNING_RATE") is not None:
        training_args.learning_rate = float(os.environ["LEARNING_RATE"])


    
    # ------ 1. Configure quantization and dtype ------
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                **training_args.bits_and_bytes_params
            )
        ))
    



    # ------ 2. Load model, tokenizer, and vision tower ------
    if args.resume_from_checkpoint:
        adapter_checkpoint = os.path.join(args.resume_from_checkpoint, 'mm_projector.bin')
        if os.path.exists(adapter_checkpoint):
            if "vision_adapter" in config.model:
                config.model.vision_adapter.checkpoint_path = adapter_checkpoint
            else:
                config.model.vision_adapter = EasyDict()
                config.model.vision_adapter.checkpoint_path = adapter_checkpoint


    tokenizer = transformers.AutoTokenizer.from_pretrained(**config.tokenizer)
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model = FridayForCausalLM.from_pretrained(
        **config.model,
        torch_dtype=compute_dtype,
        tokenizer_model_max_length=tokenizer.model_max_length,
        **bnb_model_from_pretrained_args,
    )
    model.initialize_vision_modules()


    
    
    
    # ------ 3. Configure model for training ------
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        # TODO: should the dtype be set to float32 if training_args.fp16?
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            target_modules=find_all_linear_names(model),
            task_type="CAUSAL_LM",
            **training_args.lora_params
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    
    model.get_vision_tower().to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    

    model.set_language_model_requires_grad(not training_args.freeze_language_model)
    model.set_vision_tower_requires_grad(not training_args.freeze_vision_tower)
    model.set_vision_adapter_requires_grad(not training_args.freeze_vision_adapter)

    model.print_device_configuration()


    
    

    # ------ 4. Configure Dataset and Trainer ------

    train_dataset = PretrainingDataset(
        data_path=data_args.data_path,
        image_dir=data_args.image_dir,
        tokenizer=tokenizer,
        vision_tower=model.get_vision_tower(),
        max_count=data_args.max_count,
    )

    data_collator = FridayCollator(
        tokenizer=tokenizer,
    )

    trainer = FridayTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    
    
    
    
    
    
    # ------ 5. Perform Training ------
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    rank0_print("Training completed in {:.2f} seconds".format(time.time() - start_time))
    trainer.save_state()




    # ------ 6. Save model ------
    # model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        final_checkpoint_path = os.path.join(training_args.output_dir, 'final_checkpoint')
        trainer._save(output_dir=final_checkpoint_path)
        zip_and_upload_checkpoint_artifact(
            final_checkpoint_path,
            description="Final checkpoint",
            metadata={"config": config}
        )


if __name__ == "__main__":
    train()
