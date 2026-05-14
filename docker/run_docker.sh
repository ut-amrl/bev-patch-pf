#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-bpp-training-pipeline:cuda12.8}"

usage() {
  echo "Usage:"
  echo "  $0 /path/to/your/data [/path/to/rosbagkit/config]"
  echo
  echo "Examples:"
  echo "  $0 \$HOME/data"
  echo "  $0 \$HOME/data /path/to/rosbagkit/config"
  echo
  echo "The provided host data directory will be mounted to:"
  echo "  /home/bpp/data"
  echo
  echo "The rosbagkit config directory (optional) will be mounted to:"
  echo "  /home/bpp/rosbagkit/config"
  echo "  Defaults to: <script_dir>/../../rosbagkit/config"
  echo
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Error: you must provide one or two arguments"
  echo
  usage
  exit 1
fi

HOST_DATA_DIR="$(realpath "$1")"

if [ ! -d "$HOST_DATA_DIR" ]; then
  echo "Error: data directory does not exist:"
  echo " $HOST_DATA_DIR"
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
DOCKER_CONFIG_DIR="${SCRIPT_DIR}/.docker/.config"
DOCKER_LOCAL_DIR="${SCRIPT_DIR}/.docker/.local"
if [ $# -eq 2 ]; then
  HOST_RBK_CONFIG_DIR="$(realpath "$2")"
else
  HOST_RBK_CONFIG_DIR="${SCRIPT_DIR}/../../rosbagkit/config"
fi

# create .config and .local directories if they don't exist
if [ ! -d "$DOCKER_CONFIG_DIR" ]; then
  mkdir -p "$DOCKER_CONFIG_DIR"
  echo " ${DOCKER_CONFIG_DIR} created"
fi

if [ ! -d "$DOCKER_LOCAL_DIR" ]; then
  mkdir -p "$DOCKER_LOCAL_DIR"
  echo " ${DOCKER_LOCAL_DIR} created"
fi

xhost +local:docker
# Make sure xhost privilege gets revoked
cleanup() {
  xhost -local:docker >/dev/null 2>&1 || true
}
trap cleanup EXIT

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -
chmod 777 "$XAUTH"

docker run -it --rm --gpus all \
  --runtime=nvidia \
  --privileged \
  --network=host \
  --ipc=host \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "${XSOCK}:${XSOCK}" \
  -v "${XAUTH}:${XAUTH}" \
  -e XAUTHORITY="${XAUTH}" \
  -v "${DOCKER_CONFIG_DIR}:/home/bpp/.config" \
  -v "${DOCKER_LOCAL_DIR}:/home/bpp/.local" \
  -v "${HOST_DATA_DIR}:/home/bpp/data" \
  -v "${HOST_BPP_CONFIG_DIR}:/home/bpp/bev-patch-pf/config" \
  -v "${HOST_OUTPUT_DIR}:/home/bpp/bev-patch-pf/output" \
  -v "${HOST_DATASET_DIR}:/home/bpp/bev-patch-pf/src/dataset" \
  -v "${HOST_WANDB_DIR}:/home/bpp/bev-patch-pf/wandb" \
  -v "${HOST_RBK_CONFIG_DIR}:/home/bpp/rosbagkit/config" \
  -w /home/bpp \
  --name bpp-training \
  "${IMAGE_NAME}"