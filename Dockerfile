FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

RUN apt update && \
    apt install -y \
        build-essential \
        wget \
        unzip \
        git \
        git-lfs \
        nvidia-cuda-toolkit
    

RUN git lfs install

WORKDIR /opt/ml/friday/
COPY pyproject.toml /opt/ml/friday/
RUN pip install -e . --no-build-isolation && \
    pip install --force-reinstall torchvision && \
    pip install wandb

COPY . /opt/ml/friday/
RUN pip install -e . --no-build-isolation

ENTRYPOINT ["./run.sh"]