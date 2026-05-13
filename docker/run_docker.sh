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
HOST_RBK_CONFIG_DIR="${SCRIPT_DIR}/../config"

xhost +local:docker
# Make sure xhost privilege gets revoked
cleanup() {
  xhost -local:docker >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --ipc=host added to give container all shared memory since training uses a lot of memory
# Add --net=host if problems with wandb
docker run -it --rm --gpus all \
  --ipc=host \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${HOST_DATA_DIR}:/workspace/data:rw" \
  -v "${HOST_BPP_CONFIG_DIR}:/workspace/bev-patch-pf/config:rw" \
  -v "${HOST_OUTPUT_DIR}:/workspace/bev-patch-pf/output:rw" \
  -v "${HOST_DATASET_DIR}:/workspace/bev-patch-pf/src/dataset:rw" \
  -v "${HOST_WANDB_DIR}:/workspace/bev-patch-pf/wandb:rw" \
  -v "${HOST_RBK_CONFIG_DIR}:/workspace/rosbagkit/config:rw" \
  -w /workspace \
  "${IMAGE_NAME}"
