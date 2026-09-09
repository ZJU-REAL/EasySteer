#!/usr/bin/env bash
set -euo pipefail

# Run from any directory: bash /path/to/EasySteer/docker/build.sh
# Docker inherits HTTP_PROXY/HTTPS_PROXY/NO_PROXY (or lowercase variants)
# when set. Proxy values are deliberately omitted from command arguments/logs.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

proxy_args=()
for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    if [[ -n "${!proxy_name:-}" ]]; then
        proxy_args+=(--build-arg "$proxy_name")
    fi
done

# The default official image uses CUDA 13.0. Select CUDA 12.9 with
# VLLM_BASE_IMAGE=vllm/vllm-openai:v0.29.0-cu129 bash docker/build.sh.
base_args=()
if [[ -n "${VLLM_BASE_IMAGE:-}" ]]; then
    base_args+=(--build-arg "VLLM_BASE_IMAGE=$VLLM_BASE_IMAGE")
fi
vllm_image=${VLLM_STEER_IMAGE:-vllm-steer:base}
easysteer_image=${EASYSTEER_IMAGE:-easysteer:latest}

echo "Step 1/2: Installing vllm-steer with v0.29.0 native artifacts..."
docker build \
    "${proxy_args[@]}" \
    "${base_args[@]}" \
    --target vllm-steer \
    -t "$vllm_image" \
    -f docker/Dockerfile .

echo "Step 2/2: Building EasySteer image..."
docker build "${proxy_args[@]}" "${base_args[@]}" \
    --target easysteer -t "$easysteer_image" -f docker/Dockerfile .

echo "Built $vllm_image and $easysteer_image."
echo "Run: docker run --gpus all -it $easysteer_image"
