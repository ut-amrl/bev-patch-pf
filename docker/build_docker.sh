#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-bpp-training-pipeline:cuda12.8}"
BEV_PATCH_PF_COMMIT="${BEV_PATCH_PF_COMMIT:-quattro}"
ROSBAGKIT_COMMIT="${ROSBAGKIT_COMMIT:-quattro}"

USER_UID="$(id -u)"
USER_GID="$(id -g)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -d "$SCRIPT_DIR" ]; then
  mkdir -p "${SCRIPT_DIR}/../ouput"
  echo " ${SCRIPT_DIR}/../ouput created"
fi
if [ ! -d "$SCRIPT_DIR" ]; then
  mkdir -p "${SCRIPT_DIR}/../wandb"
  echo " ${SCRIPT_DIR}/../wandb created"
fi

docker build \
    --build-arg USER_UID="${USER_UID}" \
    --build-arg USER_GID="${USER_GID}" \
    --build-arg BEV_PATCH_PF_COMMIT="${BEV_PATCH_PF_COMMIT}" \
    --build-arg ROSBAGKIT_COMMIT="${ROSBAGKIT_COMMIT}" \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}/."
