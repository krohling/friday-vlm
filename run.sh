# mkdir datasets
# wget https://friday-vlm.s3.us-west-2.amazonaws.com/LLaVA-Pretrain_small.zip
# unzip LLaVA-Pretrain_small.zip -d datasets
# rm LLaVA-Pretrain_small.zip

mkdir datasets
cd datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip -d images
rm images.zip
cd ../..

deepspeed ./friday/train/train.py --config ./config/pretrain.json

echo "***Training Complete***"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Terminating Pod"
    runpodctl remove pod $RUNPOD_POD_ID
fi