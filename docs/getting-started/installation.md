# Installation

EasySteer uses a pinned vLLM fork (`vllm-steer/`) and the `easysteer` Python
package. Choose an installation based on how you will use them:

| Use case | Installation | What it provides |
|---|---|---|
| Run the published release in a container | [Docker](#docker) | EasySteer and the matching fork/runtime; supply model weights separately. |
| Run examples in a Python environment | [Prebuilt wheel and overlay](#route-1-quick-install-prebuilt-wheel-fork-overlay) | Official kernels with the fork's Python files; no local compilation. |
| Edit the engine or EasySteer | [Development install](#route-2-development-install-recommended-for-ongoing-work) | Editable checkouts of both packages with prebuilt kernels. |

If the prebuilt artifacts do not fit your platform, use a
[source build](#fallback-build-vllm-from-source). The examples in this guide
use Linux and Python 3.12.

## Validated inference environment

The vLLM 0.29.0 steering and capture regression suites have been exercised with
this combination on Linux x86_64:

| Component | Version / hardware |
|---|---|
| Python | 3.12.13 |
| vLLM | 0.29.0, with this repository's steering fork |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.16.1 |
| CUDA runtime | 13.0.96 |
| NVIDIA driver | 580.173.02 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |

This records an exercised inference combination, not a complete compatibility
matrix or a ReFT training benchmark. Keep the official wheel's dependency pins
when installing; other hardware and package combinations need their own checks.

## Route 1: quick install (prebuilt wheel + fork overlay)

The fork's changes against upstream vLLM v0.29.0 are pure Python, so you can
install the official wheel and overlay the fork's files onto it — no build,
no editable checkouts:

```bash
conda create -n easysteer python=3.12 -y
conda activate easysteer

# Clone EasySteer with the fork commit recorded by its submodule
git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
cd EasySteer

# Official vLLM wheel (kernels prebuilt)
pip install vllm==0.29.0

# Overlay the pinned fork's Python files onto the installed package
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a vllm-steer/vllm/ "$VLLM_DIR"/

# EasySteer package
pip install .
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
cd EasySteer

# For an existing checkout, start here from the EasySteer repository root
git submodule update --init --recursive
cd vllm-steer

# EasySteer tracks the vLLM v0.29.0 release commit; pin it so the
# precompiled kernels match.
export VLLM_PRECOMPILED_WHEEL_COMMIT=98dff2a81d747d1dba01a47f939f48c3526d4206
VLLM_USE_PRECOMPILED=1 pip install --editable .

cd ..
pip install --editable .
```

Both routes finish in the EasySteer repository root, where the examples'
relative `vectors/` paths resolve.

## Fallback: build vLLM from source

Needed only when no precompiled wheel exists for your platform.

```bash
# From the EasySteer repository root
cd vllm-steer
python use_existing_torch.py

# Set your GPU architecture (e.g. "8.0" for A100) to speed up the build.
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=8.0"
export VLLM_TARGET_DEVICE="cuda"
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

pip install -r requirements/build/cuda.txt
pip install -e . --no-build-isolation -v

cd ..
pip install -e .
```

Build time depends on the selected architectures, compiler cache, and available
CPU and memory. Use the prebuilt routes when they match the target platform.

## Docker

The published v0.29.0 image includes EasySteer and its pinned vLLM fork for
Linux x86_64. Docker and NVIDIA Container Toolkit are required on the host:

```bash
docker pull xuhaolei/easysteer:v0.29.0
docker run --gpus all --shm-size=16g -it \
  -v /path/to/your/models:/app/models \
  -w /app/easysteer \
  xuhaolei/easysteer:v0.29.0
```

The version tag fixes the release; `xuhaolei/easysteer:latest` follows the
latest published version. Model weights are supplied separately through the
mounted directory.

To build from your own checkout:

```bash
# From the EasySteer repository root; requires Docker and NVIDIA Container Toolkit
bash docker/build.sh
docker run --gpus all -it \
  -v /path/to/your/models:/app/models \
  easysteer:latest
```

Here `easysteer:latest` is a local image built by the script.

The default base image uses CUDA 13.0. For a CUDA 12.9 build, select the
corresponding official image:

```bash
VLLM_BASE_IMAGE=vllm/vllm-openai:v0.29.0-cu129 bash docker/build.sh
```

Check the NVIDIA driver on the machine that will run the container. The
container supplies its CUDA runtime; it still uses the host's GPU driver.
CUDA 13.x requires driver R580 or newer. CUDA 12.x builds can support older
drivers, but JIT/PTX compatibility must also be checked on the target GPU;
see NVIDIA's [compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## CUDA library discovery

If installation succeeds but the first generation fails with
`ld: cannot find -lcudart`, check the toolkit path used for JIT compilation.
`CUDA_HOME` must point to a toolkit whose library directory is discoverable by
the compiler and contains the link name `libcudart.so`. Some pip-provided CUDA
layouts contain only a versioned library such as `libcudart.so.13`, which is not
enough for the linker's `-lcudart` lookup.

Keep an already working system toolkit configuration unless the chosen package
build requires a different one. A missing linker path or link name does not by
itself establish a driver/runtime version incompatibility.
