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
git clone https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data
rm -rf Bunny-v1_0-data/.git
cat Bunny-v1_0-data/pretrain/images.tar.gz.part-* > Bunny-v1_0-data/pretrain/images.tar.gz
cd Bunny-v1_0-data/pretrain
tar -xvzf images.tar.gz
rm images.tar.gz
cd ../..
pytest -s
# pip install wandb
# wandb login <your_api_key>
# export WANDB_ENTITY=
# export WANDB_PROJECT=
# deepspeed ./friday/train/train.py --config ./friday/train/config/pretrain.json