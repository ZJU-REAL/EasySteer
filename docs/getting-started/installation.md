# Installation

EasySteer ships as two packages installed from one repository: the vLLM fork
(`vllm-steer/`) and the `easysteer` Python package.

## Recommended: precompiled vLLM wheel

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer/vllm-steer

# EasySteer tracks the vLLM v0.26.0 release commit; pin it so the
# precompiled kernels match.
export VLLM_PRECOMPILED_WHEEL_COMMIT=568afb3a13806beb53bb2e6bd518269357b237c0
VLLM_USE_PRECOMPILED=1 pip install --editable .

cd ..
pip install --editable .
```

## Fallback: build vLLM from source

Needed only when no precompiled wheel exists for your platform.

```bash
cd EasySteer/vllm-steer
python use_existing_torch.py

# Set your GPU architecture (e.g. "8.0" for A100) to speed up the build.
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=8.0"
export VLLM_TARGET_DEVICE="cuda"
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

pip install -r requirements/build.txt
pip install -e . --no-build-isolation -v

cd ..
pip install -e .
```

A full source build can take from ~20 minutes (128 cores) to several hours.

## Docker

!!! note
    The published image (`xuhaolei/easysteer`, tag `v0.17.1`) predates the
    vLLM v0.26.0 migration — it runs the previous engine and v1-era APIs.
    A refreshed image is planned; until then, prefer the wheel install
    above for current features.

```bash
docker pull xuhaolei/easysteer:latest
docker run --gpus all -it \
  -v /path/to/your/models:/app/models \
  easysteer:latest
python3 /app/easysteer/docker/docker_test.py
```

<!-- TODO: verify supported Python/CUDA version matrix and document it here. -->
