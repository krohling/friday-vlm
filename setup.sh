mkdir -p ~/miniconda3
apt update
apt install nvidia-cuda-toolkit git-lfs -y
git lfs install
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda create --name friday python=3.12 -y
conda activate friday
conda install pytorch torchvision -c pytorch -y
git clone https://github.com/krohling/friday-vlm.git
cd friday-vlm
# pip install -e torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 -c constraints.txt
# pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121 -c constraints.txt
pip install -e . --no-build-isolation
pip install --force-reinstall torchvision
pip install wandb

mkdir datasets
# wget https://friday-vlm.s3.us-west-2.amazonaws.com/LLaVA-Pretrain_small.zip
# unzip LLaVA-Pretrain_small.zip -d datasets
# rm LLaVA-Pretrain_small.zip

wget https://friday-vlm.s3.us-west-2.amazonaws.com/llava_v1_5_mix665k_small.zip
unzip llava_v1_5_mix665k_small.zip -d datasets
rm llava_v1_5_mix665k_small.zip

wget https://friday-vlm.s3.us-west-2.amazonaws.com/llava_v1_5_mix665k.zip
unzip llava_v1_5_mix665k.zip -d datasets
rm llava_v1_5_mix665k.zip

pytest -s

# wandb login <your_api_key>
# export WANDB_ENTITY=
# export WANDB_PROJECT=
# deepspeed ./friday/train/train.py --config ./friday/train/config/pretrain.json