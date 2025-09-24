# Stage 1: Base Stage
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 as dev-git
# FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Step 2: Install system dependencies, including Python and git.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git libglib2.0-0 wget curl htop nvtop vim && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Install uv, our fast package manager.
COPY --from=ghcr.io/astral-sh/uv:0.8.20 /uv /bin/uv

WORKDIR /workspaces

# Step 4: Install all Python packages using uv.

# Create a virtual environment with uv
RUN uv venv /venv

# Activate venv for all shells and processes
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

RUN uv pip install --no-cache-dir setuptools_scm
RUN uv pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.8.0 torchvision torchaudio==2.8.0
    # --index-url https://download.pytorch.org/whl/cu128 \
    # torch==2.7.1 torchvision torchaudio==2.7.1

RUN uv pip install --no-cache-dir --no-build-isolation flash-attn

RUN uv pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/huggingface/accelerate \
    git+https://github.com/huggingface/peft \
    git+https://github.com/huggingface/trl \
    git+https://github.com/EleutherAI/lm-evaluation-harness.git \
    bitsandbytes \
    deepspeed \
    liger-kernel

# RUN python3 -m pip install liger-kernel-nightly

RUN rm -rf /root/.cache

# # Stage 2: Development with Git
#     datasets numpy sentencepiece gguf protobuf matplotlib \
#     bitsandbytes \
#     liger-kernel \
#     tensorboard
#     # lm_eval

# RUN python3 -m pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
# # RUN python3 -m pip install llmcompressor
