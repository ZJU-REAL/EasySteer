# syntax=docker/dockerfile:1.19
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${VLLM_BASE_IMAGE} AS vllm-steer

WORKDIR /app/easysteer/vllm-steer
COPY vllm-steer/requirements/build/cuda.txt /tmp/vllm-build-requirements.txt
# Keep the CUDA-specific torch package supplied by the selected official image.
RUN python3 -c 'from importlib.metadata import version; print("torch==" + version("torch"))' \
        > /opt/vllm-constraints.txt \
    && python3 -m pip install --no-cache-dir -c /opt/vllm-constraints.txt \
        -r /tmp/vllm-build-requirements.txt

COPY vllm-steer/ ./
# Upstream setup restores native libraries, Rust and generated Python interfaces
# from the release wheel matching the base image's torch CUDA version.
RUN VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=98dff2a81d747d1dba01a47f939f48c3526d4206 \
    VLLM_VERSION_OVERRIDE=0.29.0+easysteer \
    VLLM_REQUIRE_RUST_FRONTEND=1 \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e .

RUN cd /tmp && python3 - <<'PY'
import subprocess
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

source = Path("/app/easysteer/vllm-steer/vllm")
assert Path(find_spec("vllm").origin).parent == source
assert version("vllm") == "0.29.0+easysteer"
for pattern in (
    "_C*.so",
    "_moe_C*.so",
    "_rust_tool_parser*.so",
    "vllm-rs",
    "third_party/flashmla/flash_mla_interface.py",
):
    assert any(source.glob(pattern)), f"Missing precompiled artifact: {pattern}"
subprocess.run([source / "vllm-rs", "--version"], check=True)
PY

FROM vllm-steer AS easysteer

LABEL maintainer="ZJU-REAL"
LABEL description="EasySteer - High-Performance LLM Steering Framework"

WORKDIR /app/easysteer
# Do not overwrite Python interfaces that setup extracted into vllm-steer.
COPY --exclude=vllm-steer . ./
RUN python3 -m pip install --no-cache-dir -c /opt/vllm-constraints.txt -e .

ENV PYTHONUNBUFFERED=1
RUN mkdir -p /app/workspace
WORKDIR /app/workspace
ENTRYPOINT []
CMD ["/bin/bash"]
