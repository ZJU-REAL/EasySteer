---
title: EasySteer Demo
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# EasySteer Space

A Gradio demo for comparing baseline and steered responses with the bundled
Qwen2.5-1.5B-Instruct vectors.

## API mode (CPU)

From this directory:

```bash
docker build --build-arg MODE=api -t easysteer-space:api .
docker run --rm -p 7860:7860 \
  -e VLLM_API_URL=http://your-inference-host:8000/v1 \
  -e VLLM_MODEL_NAME=your-served-model \
  -e VLLM_VECTOR_BASE_PATH=/absolute/server/path/to/hf-space \
  easysteer-space:api
```

In a Hugging Face Docker Space, configure these values under Settings →
Variables; put an optional `VLLM_API_KEY` in Secrets. The URL must be reachable
from the Space container. `DEMO_MODE=api` requires the URL and model name and
reports missing settings immediately. It does not fall back to loading a GPU
model. Direct Python launches default to the same mode.

Run a vllm-steer server using Qwen2.5-1.5B-Instruct and declare the algorithms
needed by these presets, for example `--enable-steer-vector
--steer-algorithms direct,loreft --steer-multi-vector --enforce-eager`.
See the [server guide](https://github.com/ZJU-REAL/EasySteer/blob/main/docs/user-guide/openai-server.md) for installation
and full startup examples. Copy the GGUF files in `vectors/` to that server;
`VLLM_VECTOR_BASE_PATH` prefixes their paths in requests. The bundled GGUF
files contain the actual vector data and are included in this checkout.

The bundled LoReFT preset sends its exported `payload.json` as canonical
steering data, so the API image needs neither torch nor vLLM. This JSON stays
on the Space; it is not looked up under `VLLM_VECTOR_BASE_PATH`. Its layer and
hidden size are specific to the bundled model.

## Presets

Presets in `configs/` contain an `instruction`, numeric `sampling` settings,
and a native `steering` spec. For example:

```json
{
  "instruction": "Alice's dog has passed away. Please comfort her.",
  "sampling": {
    "temperature": 0.0,
    "max_tokens": 128,
    "repetition_penalty": 1.1
  },
  "steering": {
    "vectors": [{
      "source": "vectors/happy_diffmean.gguf",
      "algorithm": "direct",
      "scale": 2.0,
      "layers": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
      "normalize": false,
      "apply": {"prompt": "all", "generation": "all"}
    }]
  }
}
```

Each vector uses the public `source`, `algorithm`, `scale`, `layers`,
`normalize`, and `apply` fields. Add vectors to the same list for multi-vector
steering; set `steering.conflict` when needed. The bundled refusal preset
uses `"conflict": "sequential"` and applies its four vectors at prompt
positions `[-1]`, `[-2]`, `[-3]`, and `[-4]`. An apply clause such as
`{"prompt_positions": [-1]}` affects only the last prompt position.

Before a request, the demo resolves `source` paths and loads any local
`payload_path` into `data`. The resulting spec is sent directly to the API.
The model comes from the deployment settings above.

## Exporting other checkpoints

Checkpoint-based algorithms (`linear`, `lm_steer`, `loreft`) need a local JSON
payload in API mode. In an environment with EasySteer and vllm-steer installed:

```bash
python hf-space/export_payload.py /path/to/checkpoint \
  /path/to/payload.json --algorithm loreft
```

Set `payload_path` in the appropriate `steering.vectors` entry to the exported
file, relative to `hf-space/`, and include it in the image. For example, the
bundled LoReFT vector uses:

```json
{
  "payload_path": "results/emoji_loreft/payload.json",
  "algorithm": "loreft",
  "scale": 1.0,
  "layers": [22],
  "apply": {"prompt_positions": [-1]}
}
```

`payload_path` is the demo's file reference for exported payloads; it is
replaced with canonical `data` before sending the spec. Export reads trusted
checkpoint files using the public EasySteer adapter; Git LFS pointers must
first be replaced with the actual weights.

CPU checks (run from the repository root):

```bash
python -m pip install -r hf-space/requirements.txt pytest pybase64
python -m pytest hf-space/tests -q
```

## GPU mode

First build an EasySteer image from the current checkout with
`bash docker/build.sh`. Then, from `hf-space/`:

```bash
docker build --build-arg MODE=gpu --build-arg GPU_BASE=easysteer:latest \
  -t easysteer-space:gpu .
docker run --rm --gpus all -p 7860:7860 \
  -e EASYSTEER_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  easysteer-space:gpu
```

The build sets `DEMO_MODE=gpu`. Use a model ID or mount a local model directory
and set `EASYSTEER_MODEL` to its container path. The bundled presets expect
Qwen2.5-1.5B-Instruct; changing only the model name does not adapt the vectors.
GPU mode runs an eager local engine and uses the same exported payloads when
present.
