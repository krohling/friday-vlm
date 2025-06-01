FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN pip install --no-build-isolation \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl


RUN apt update && \
    apt install -y \
        build-essential \
        wget \
        unzip \
        git \
        git-lfs
    

RUN git lfs install

WORKDIR /opt/ml/friday/
COPY pyproject.toml /opt/ml/friday/
RUN pip install -e . --no-build-isolation && \
    pip install --force-reinstall torchvision && \
    pip install wandb

COPY . /opt/ml/friday/

ENTRYPOINT ["./run.sh"]