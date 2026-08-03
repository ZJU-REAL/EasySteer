# Installation

EasySteer ships as two packages installed from one repository: the vLLM fork
(`vllm-steer/`) and the `easysteer` Python package. Pick one of two routes:

- **Quick install** — stock vLLM wheel plus a file overlay of the fork's
  Python changes. The fastest way to a working environment; not editable.
- **Development install** — editable checkouts of both packages; changes to
  the fork or to `easysteer` take effect immediately. Use this if you plan to
  develop, debug, or track the repository.

## Route 1: quick install (prebuilt wheel + fork overlay)

The fork's changes against upstream vLLM v0.26.0 are pure Python, so you can
install the official wheel and overlay the fork's files onto it — no build,
no editable checkouts:

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

# Official vLLM wheel (kernels prebuilt)
pip install vllm==0.26.0

# Overlay the fork's Python files onto the installed package
git clone --depth 1 https://github.com/ZJU-REAL/EasySteer-vllm-v1.git
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a EasySteer-vllm-v1/vllm/ "$VLLM_DIR"/

# EasySteer package
git clone https://github.com/ZJU-REAL/EasySteer.git
pip install ./EasySteer
```

!!! warning
    The overlay is not tracked by pip: reinstalling or upgrading `vllm`
    silently reverts it (re-run the rsync afterwards), and `pip show vllm`
    still reports the stock package. For anything long-lived, prefer Route 2.

## Route 2: development install (recommended for ongoing work)

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
