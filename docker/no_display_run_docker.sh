#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-bpp-training-pipeline:cuda12.8}"

usage() {
  echo "Usage:"
  echo "  $0 /path/to/your/data"
  echo
  echo "Example:"
  echo "  $0 \$HOME/data"
  echo
  echo "The provided host data directory will be mounted to:"
  echo "  /workspace/data"
  echo
}

if [ $# -ne 1 ]; then
  echo "Error: you must provide one argument that is the host data directory"
  echo
  usage
  exit 1
fi

HOST_DATA_DIR="$(realpath "$1")"

if [ ! -d "$HOST_DATA_DIR" ]; then
  echo "Error: data directory does not exist:"
  echo " ${HOST_DATA_DIR}"
  echo
  echo "Create it first"
  echo
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# paths relative to bev-patch-pf/docker
HOST_BPP_CONFIG_DIR="${SCRIPT_DIR}/../config"
HOST_OUTPUT_DIR="${SCRIPT_DIR}/../output"
HOST_DATASET_DIR="${SCRIPT_DIR}/../src/dataset"
HOST_WANDB_DIR="${SCRIPT_DIR}/../wandb"
HOST_PAPER_DIR="${SCRIPT_DIR}/../paper_scripts"
HOST_RBK_CONFIG_DIR="${SCRIPT_DIR}/../config"


# --ipc=host added to give container all shared memory since training uses a lot of memory
# Add --net=host if problems with wandb
docker run -it --rm --gpus all \
  --ipc=host \
  -e QT_X11_NO_MITSHM=1 \
  -v "${HOST_DATA_DIR}:/home/bpp/data" \
  -v "${HOST_BPP_CONFIG_DIR}:/home/bpp/bev-patch-pf/config" \
  -v "${HOST_OUTPUT_DIR}:/home/bpp/bev-patch-pf/output" \
  -v "${HOST_DATASET_DIR}:/home/bpp/bev-patch-pf/src/dataset" \
  -v "${HOST_WANDB_DIR}:/home/bpp/bev-patch-pf/wandb" \
  -v "${HOST_PAPER_DIR}:/home/bpp/bev-patch-pf/paper_scripts" \
  -v "${HOST_RBK_CONFIG_DIR}:/home/bpp/rosbagkit/config" \
  -w /home/bpp \
  --name bpp-training \
  "${IMAGE_NAME}"