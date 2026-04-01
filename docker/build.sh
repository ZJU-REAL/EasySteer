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

# Temporarily patch Dockerfile:
# 1) Override version string
sed -i.bak 's/ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+csrc.build"/ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.17.1+easysteer"/' docker/Dockerfile
# 2) Set version in the 'build' stage (csrc-build already has it from step 1)
sed -i '/^FROM base AS build/a ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.17.1+easysteer"' docker/Dockerfile
# 3) Use Chinese Ubuntu mirrors to avoid proxy 502 errors
sed -i '/^RUN apt-get update/i RUN sed -i "s|http://archive.ubuntu.com|http://mirrors.aliyun.com|g; s|http://security.ubuntu.com|http://mirrors.aliyun.com|g" /etc/apt/sources.list' docker/Dockerfile

docker build \
  ${PROXY_ARGS} \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg VLLM_USE_PRECOMPILED=1 \
  --build-arg VLLM_MERGE_BASE_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995 \
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
