FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/bev-patch-pf/bin:/opt/conda/bin:$PATH

# System deps:
# - libgl1/libglib2.0-0 are commonly needed by opencv-python
# - bash/ca-certificates: convenience and HTTPS certs
# - vim/nano included for editing yaml files
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    vim \
    nano \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=bpp
ARG USER_UID=1000
ARG USER_GID=1000

# Create or reuse a group/user for the requested UID/GID.
#
# This avoids build failures when:
# - USER_GID already exists in the base image
# - USER_UID already exists in the base image
#
# If the UID already exists, we rename that user to ${USERNAME}.
# If the GID already exists, we reuse that group.
RUN set -eux; \
    if getent group "${USER_GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        GROUP_NAME="${USERNAME}"; \
        groupadd --gid "${USER_GID}" "${GROUP_NAME}"; \
    fi; \
    if getent passwd "${USER_UID}" >/dev/null; then \
        EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        if [ "${EXISTING_USER}" != "${USERNAME}" ]; then \
            usermod -l "${USERNAME}" "${EXISTING_USER}"; \
            usermod -d "/home/${USERNAME}" -m "${USERNAME}" || true; \
        fi; \
        usermod -g "${USER_GID}" "${USERNAME}"; \
        usermod -s /bin/bash "${USERNAME}"; \
    else \
        useradd \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --create-home \
            --shell /bin/bash \
            "${USERNAME}"; \
    fi

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

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone bev-patch-pf repo.
ARG BEV_PATCH_PF_REPO=https://github.com/ut-amrl/bev-patch-pf.git
ARG BEV_PATCH_PF_COMMIT=main
RUN git clone "${BEV_PATCH_PF_REPO}" /workspace/bev-patch-pf \
    && cd /workspace/bev-patch-pf \
    && git checkout "${BEV_PATCH_PF_COMMIT}"

RUN pip install --no-cache-dir /workspace/bev-patch-pf

# multi gpu capable
RUN pip install --no-cache-dir accelerate

# for some reason `onnxscript` does not get installed with the other onnx dependencies
RUN pip install --no-cache-dir onnxscript

# Make data direcotry
RUN mkdir -p /workspace/bev-patch-pf/data \
    && chown -R "${USER_UID}:${USER_GID}" /workspace

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

USER ${USERNAME}

CMD ["/bin/bash"]