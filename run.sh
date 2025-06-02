#!/bin/bash

mkdir datasets
wget https://friday-vlm.s3.us-west-2.amazonaws.com/llava_v1_5_mix665k_small.zip
unzip llava_v1_5_mix665k_small.zip -d datasets
rm llava_v1_5_mix665k_small.zip

PYTHONPATH=. deepspeed friday/train/train.py --config ./config/finetune.json


echo "***Training Complete***"
if [ -n "$RUNPOD_POD_ID" ]; then
    sleep 60
    echo "Terminating Pod"
    runpodctl remove pod $RUNPOD_POD_ID
fi