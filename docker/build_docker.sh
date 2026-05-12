#!/usr/bin/env bash
set -euo pipefail

# Run from within bev-patch-pf/docker

IMAGE_NAME="${IMAGE_NAME:-bpp-training-pipeline:cuda12.8}"
BEV_PATCH_PF_COMMIT="${BEV_PATCH_PF_COMMIT:-quattro}"

USER_UID="$(id -u)"
USER_GID="$(id -g)"

docker build \
    --build-arg USER_UID="${USER_UID}" \
    --build-arg USER_GID="${USER_GID}" \
    --build-arg BEV_PATCH_PF_COMMIT="${BEV_PATCH_PF_COMMIT}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile \
    .
