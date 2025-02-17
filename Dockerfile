# Stage 1: Base Stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel as base

# Install Python and other necessary packages
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch torchvision torchaudio
WORKDIR /workspaces


# Stage 2: Development with Git
FROM base as dev-git
RUN python3 -m pip install git+https://github.com/huggingface/transformers \
    git+https://github.com/huggingface/accelerate \
    git+https://github.com/huggingface/peft \
    git+https://github.com/huggingface/trl \
    flash-attn --no-build-isolation \
    datasets numpy sentencepiece gguf protobuf matplotlib \
    bitsandbytes \
    liger-kernel \
    llama-recipes \
    ollama \
    tensorboard \
    qwen-vl-utils[decord]
# RUN python3 -m pip install liger-kernel-nightly
# RUN python3 -m pip install pymysql

# Stage 3: Development with Stable Packages
FROM base as dev-stable
RUN python3 -m pip install transformers accelerate peft trl \
    flash-attn --no-build-isolation \
    datasets numpy sentencepiece gguf protobuf matplotlib \
    bitsandbytes \
    liger-kernel \
    llama-recipes \
    ollama \
    tensorboard \
    qwen-vl-utils[decord]

