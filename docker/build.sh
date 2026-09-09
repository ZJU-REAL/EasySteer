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

# Use a separate Dockerfile so a failed/interrupted build cannot modify
# the submodule checkout. The source uses native artifacts from vLLM v0.28.0.
build_dockerfile=$(mktemp "${TMPDIR:-/tmp}/easysteer-dockerfile.XXXXXX")
trap 'rm -f "$build_dockerfile"' EXIT
awk '
    /^ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0\+csrc.build"$/ {
        print "ENV SETUPTOOLS_SCM_PRETEND_VERSION=\"0.28.0+easysteer\""
        version_count++
        next
    }
    { print }
    /^FROM base AS build$/ {
        print "ENV SETUPTOOLS_SCM_PRETEND_VERSION=\"0.28.0+easysteer\""
        build_count++
    }
    END { if (version_count != 1 || build_count != 1) exit 1 }
' vllm-steer/docker/Dockerfile > "$build_dockerfile"

echo "Step 1/2: Building vllm-steer base image (v0.28.0 native artifacts)..."
docker build \
    "${proxy_args[@]}" \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg VLLM_USE_PRECOMPILED=1 \
    --build-arg VLLM_MERGE_BASE_COMMIT=2cf0a6915ce544dc493a0990f2ea38d81601128a \
    --build-arg GIT_REPO_CHECK=0 \
    --target vllm-openai \
    -t vllm-steer:base \
    -f "$build_dockerfile" \
    vllm-steer

echo "Step 2/2: Building EasySteer image..."
docker build "${proxy_args[@]}" -t easysteer:latest -f docker/Dockerfile .

echo "Built vllm-steer:base and easysteer:latest."
echo "Run: docker compose -f docker/docker-compose.yml up -d"
echo "Or:  docker run --gpus all -it easysteer:latest"
