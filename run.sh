mkdir datasets
wget https://friday-vlm.s3.us-west-2.amazonaws.com/LLaVA-Pretrain_small.zip
unzip LLaVA-Pretrain_small.zip -d datasets
rm LLaVA-Pretrain_small.zip

deepspeed ./friday/train/train.py --config ./config/test.json

echo "***Training Complete***"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Terminating Pod"
    runpodctl remove pod $RUNPOD_POD_ID
fi