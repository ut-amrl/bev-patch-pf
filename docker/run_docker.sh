#!/usr/bin/env bash
set -e

IMAGE_NAME="bpp-training-pipeline:cuda12.8"

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/host/data-dir"
  exit 1
fi

HOST_DATA_DIR="$(realpath "$1")"

if [ ! -d "$HOST_DATA_DIR" ]; then
  echo "Error: data directory does not exist: $HOST_DATA_DIR"
  exit 1
fi

docker run --rm -it --gpus all \
  -v "$HOST_DATA_DIR":/workspace/bev-patch-pf/data \
  -w /workspace/bev-patch-pf \
  "$IMAGE_NAME"
