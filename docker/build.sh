#!/bin/bash
set -e

# EasySteer Docker Build Script
#
# Usage:
#   bash build.sh                    # Build without proxy
#   http_proxy=... bash build.sh     # Build with proxy
#
# Environment Variables (optional):
#   http_proxy / HTTP_PROXY          # HTTP proxy URL
#   https_proxy / HTTPS_PROXY        # HTTPS proxy URL

echo "Building EasySteer Docker images..."

# Navigate to project root
cd "$(dirname "$0")/.."

# Prepare proxy arguments (only if set)
PROXY_ARGS=""
if [ ! -z "${http_proxy}" ] || [ ! -z "${HTTP_PROXY}" ]; then
    PROXY_ARGS="--build-arg HTTP_PROXY=${HTTP_PROXY:-${http_proxy}} \
                --build-arg http_proxy=${http_proxy:-${HTTP_PROXY}}"
    echo "Using HTTP proxy: ${http_proxy:-${HTTP_PROXY}}"
fi
if [ ! -z "${https_proxy}" ] || [ ! -z "${HTTPS_PROXY}" ]; then
    PROXY_ARGS="${PROXY_ARGS} \
                --build-arg HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy}} \
                --build-arg https_proxy=${https_proxy:-${HTTPS_PROXY}}"
    echo "Using HTTPS proxy: ${https_proxy:-${HTTPS_PROXY}}"
fi

# Step 1: Build vllm-steer base image
echo "Step 1/2: Building vllm-steer base image (with precompiled wheel)..."
cd vllm-steer

# Temporarily patch Dockerfile to add SETUPTOOLS_SCM_PRETEND_VERSION
sed -i.bak '280a\
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.13.0+easysteer"
' docker/Dockerfile

docker build \
  ${PROXY_ARGS} \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg VLLM_USE_PRECOMPILED=1 \
  --build-arg VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502 \
  --build-arg GIT_REPO_CHECK=0 \
  --target vllm-openai \
  -t vllm-steer:base \
  -f docker/Dockerfile \
  .

# Restore original Dockerfile
mv docker/Dockerfile.bak docker/Dockerfile 2>/dev/null || true

# Step 2: Build EasySteer on top
echo "Step 2/2: Building EasySteer image..."
cd ..
docker build \
  ${PROXY_ARGS} \
  -t easysteer:latest \
  -f docker/Dockerfile \
  .

echo "Build complete!"
echo ""
echo "Images created:"
echo "  - vllm-steer:base"
echo "  - easysteer:latest"
echo ""
echo "Run with: docker-compose -f docker/docker-compose.yml up -d"
echo "Or: docker run --gpus all -it easysteer:latest"
