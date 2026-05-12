FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/bev-patch-pf/bin:/opt/conda/bin:$PATH

# Basic system dependencies.
# libgl1/libglib2.0-0 are commonly needed by opencv-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba.
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create the conda environment and pin numpy to match manifpy.
RUN micromamba create -y -n bev-patch-pf -c conda-forge \
    python=3.12 \
    pip \
    numpy=1.26.4 \
    manifpy \
    && micromamba clean -a -y

# Make future RUN commands use the environment.
SHELL ["micromamba", "run", "-n", "bev-patch-pf", "/bin/bash", "-c"]

WORKDIR /workspace

RUN pip install --upgrade pip setuptools wheel

# Clone bev-patch-pf repo.
ARG BEV_PATCH_PF_COMMIT=c26f10120e566809f28e54b0e2cee8d9fa7ee10a
RUN git clone https://github.com/ut-amrl/bev-patch-pf.git /workspace/bev-patch-pf \
    && cd /workspace/bev-patch-pf \
    && git checkout ${BEV_PATCH_PF_COMMIT}

RUN pip install /workspace/bev-patch-pf

# Make data direcotry
RUN mkdir -p /workspace/bev-patch-pf/data

# Sanity-check imports at build time.
WORKDIR /workspace/bev-patch-pf

RUN python - <<'EOF'
import numpy
import torch
import manifpy
import onnx
import rasterio
import hydra
import huggingface_hub
EOF

CMD ["/bin/bash"]