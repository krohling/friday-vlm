#!/bin/bash

mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip -d images
rm images.zip
cd ../..

PYTHONPATH=. deepspeed friday/train/train.py --config ./config/pretrain.json


echo "***Training Complete***"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Terminating Pod"
    runpodctl remove pod $RUNPOD_POD_ID
fi