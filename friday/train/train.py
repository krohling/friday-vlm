import argparse
import os

import json
from easydict import EasyDict

import logging
import pathlib

import torch

import transformers

from friday.train.friday_trainer import FridayTrainer

from friday import conversation as conversation_lib
from friday.model import *
from friday.util.data_utils import make_supervised_data_module
from friday.train.config import FridayTrainingArguments, DataArguments

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.util.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank

    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument('--config', type=str, help='The path to the config json file')
    parser.add_argument('--deepspeed', type=str, help='The path to the deepspeed config json file')
    parser.add_argument('--local_rank', type=int, help='The local rank for distributed training', default=-1)
    args = parser.parse_args()
    local_rank = args.local_rank

    # ------ 0. Load config ------
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    with open(args.config, 'r') as f:
        config = EasyDict(json.load(f))
    
    training_args = FridayTrainingArguments(**config.training)
    training_args.deepspeed = args.deepspeed if args.deepspeed else None
    

    
    
    

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
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model = FridayForCausalLM.from_pretrained(
        config.model.language_model.model_name_or_path,
        cache_dir=training_args.cache_dir,
        cfg_vision_tower=config.model.vision_tower,
        cfg_vision_adapter=config.model.vision_adapter,
        torch_dtype=compute_dtype,
        **bnb_model_from_pretrained_args,
        **config.model.language_model.model_params,
    )
    model.initialize_vision_modules()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.language_model.tokenizer_name_or_path,
        cache_dir=training_args.cache_dir,
        **config.model.language_model.tokenizer_params,
    )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token


    
    
    
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
    

    if config.model.language_model.freeze:
        model.set_language_model_requires_grad(False)
    if config.model.vision_tower.freeze:
        model.set_vision_tower_requires_grad(False)
    if config.model.vision_adapter.freeze:
        model.set_vision_adapter_requires_grad(False)

    model.print_device_configuration()
    
    
    

    # model.config.image_aspect_ratio = data_args.image_aspect_ratio
    # model.config.tokenizer_padding_side = tokenizer.padding_side
    # model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    # model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    # model.config.mm_projector_lr = training_args.mm_projector_lr
    # model.config.use_s2 = model_args.use_s2

    # model.config.unfreeze_vision_tower = training_args.unfreeze_vision_tower = model_args.unfreeze_vision_tower


    
    



    # ------ 5. Configure conversation and data module ------

    from friday.train.data import PretrainingDataset, PretrainingCollator

    # conversation_lib.default_conversation = conversation_lib.conv_templates["default"]

    # data_module = make_supervised_data_module(tokenizer=tokenizer,
    #                                           vision_tower=model.get_vision_tower(),
    #                                           data_args=config.data)

    train_dataset = PretrainingDataset(
        data_path=config.data.data_path,
        image_dir=config.data.image_dir,
        tokenizer=tokenizer,
        vision_tower=model.get_vision_tower(),
        max_count=config.data.max_count,
    )

    data_collator = PretrainingCollator(
        tokenizer=tokenizer,
    )
    

    trainer = FridayTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    
    
    
    
    
    
    
    
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

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
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
