#!/bin/bash

MODEL_TYPE=phi-4
OUTPUT_DIR=friday-$MODEL_TYPE-pretrain

mkdir -p ./checkpoints-pretrain/$OUTPUT_DIR

deepspeed friday/train/train.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path microsoft/Phi-4-mini-instruct \
    --model_type $MODEL_TYPE \
    --version plain \
    --data_path ./Bunny-v1_0-data/pretrain/bunny_pretrain_laion_2m.json \
    --image_folder ./Bunny-v1_0-data/pretrain/images/ \
    --vision_tower google/siglip2-base-patch16-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --image_aspect_ratio square \
    --bf16 True \
    --output_dir ./checkpoints-pretrain/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-pretrain/$OUTPUT_DIR/log.txt
    # --evaluation_strategy "no" \
