mkdir -p ~/miniconda3
apt install nvidia-cuda-toolkit -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda create --name friday python=3.12 -y
conda activate friday
conda install pytorch torchvision -c pytorch -y
git clone https://github.com/krohling/friday-vlm.git
cd friday-vlm
pip install -e .
# pip install --force-reinstall torchvision
