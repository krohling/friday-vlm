import argparse
import os
import time

import json
from easydict import EasyDict

import torch

import transformers

from model_factory import build_model
from friday.model import *
from friday.data import PretrainingDataset, FinetuningDataset, FridayCollator
from friday.train.friday_trainer import FridayTrainer, zip_and_upload_checkpoint_artifact
from friday.train.config import FridayTrainingArguments, FridayDataArguments
from friday.util import (
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
    parser.add_argument('--mm_projector_checkpoint', type=str, help='The checkpoint to load for the vision adapter', default=None)
    args = parser.parse_args()
    local_rank = args.local_rank

    # ------ 0. Load config ------
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    with open(args.config, 'r') as f:
        config = EasyDict(json.load(f))
    
    print(f"local_rank: {local_rank}")
    print(config)

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

    
    # ------ 1. Build the model and tokenizer ------
    
    model, tokenizer = build_model(
        model_config=config.model,
        tokenizer_config=config.tokenizer,
        training_args=training_args,
        mm_projector_checkpoint=args.mm_projector_checkpoint,
    )
    if local_rank in [0, -1]:
        model.print_device_configuration()

    
    

    # ------ 2. Configure Dataset and Trainer ------

    if data_args.dataset_type == "finetuning":
        print("Using finetuning dataset")
        train_dataset = FinetuningDataset(
            data_path=data_args.data_path,
            image_dir=data_args.image_dir,
            tokenizer=tokenizer,
            vision_tower=model.get_vision_tower(),
            max_count=data_args.max_count,
        )
    else:
        print("Using pretraining dataset")
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

    
    
    
    
    
    
    # ------ 3. Perform Training ------
    start_time = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    rank0_print("Training completed in {:.2f} seconds".format(time.time() - start_time))
    trainer.save_state()




    # ------ 4. Save model ------
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if local_rank in [0, -1]:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        final_checkpoint_path = os.path.join(training_args.output_dir, 'final_checkpoint')
        trainer._save(output_dir=final_checkpoint_path)
        if "wandb" in getattr(training_args, "report_to", []):
            zip_and_upload_checkpoint_artifact(
                final_checkpoint_path,
                description="Final checkpoint",
                metadata={"config": config}
            )


if __name__ == "__main__":
    train()
