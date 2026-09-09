# EasySteer Frontend

The Vue UI sends generation requests to a separate vllm-steer
OpenAI-compatible server. The Flask job backend handles vector extraction,
ReFT training, and SAE exploration. A built UI and `/api` are served from
one origin so Workshop and SAE requests reach the job backend.

![EasySteer Frontend](../figures/demosys.png)

## Setup

First install EasySteer and vllm-steer using the
[installation guide](../docs/getting-started/installation.md). Use that
Python environment for the job backend. From the repository root:

```bash
python -m pip install -r frontend/requirements.txt
cd frontend/app
npm ci
npm run build
cd ../..
bash frontend/start.sh
```

Open `http://localhost:8111`. The script also binds port 5000 for direct
job API clients. Both ports serve the same application. Set
`EASYSTEER_FRONTEND_PORT` and `EASYSTEER_BACKEND_PORT` to change them; set both
to the same value to use one port. `EASYSTEER_HOST` defaults to `127.0.0.1`;
set it to `0.0.0.0` when exposing the demo through your deployment.
`EASYSTEER_PYTHON=/path/to/python` selects the installed environment.
The script works from any directory and does not install packages at startup.
Gunicorn runs one worker with threads because job status and model instances
are held in process memory. Ctrl+C stops the server and its worker.

Start a vllm-steer server separately, declaring the algorithms you plan to
use, as shown in the [server guide](../docs/user-guide/openai-server.md).
In the UI, set its base URL (default `http://localhost:8000/v1`) and served
model name. The URL must be reachable by your browser; a remote deployment
may require a reverse proxy or server CORS configuration. Extraction and
training model paths are resolved by the job backend; inference vector paths
are resolved by vllm-steer. Use matching model weights and compatible vectors.

Extraction requires the V2 model runner: the Qwen2.5 presets use it by default;
for models that default to V1, set `VLLM_USE_V2_MODEL_RUNNER=1` before starting
the job backend. See the [capture guide](../docs/user-guide/hidden-state-capture.md).

## Development

Run these commands from the repository root in separate terminals:

```bash
# Job backend; add FLASK_DEBUG=1 only when debugging locally.
python frontend/app.py
```

```bash
cd frontend/app
npm ci
npm run dev
```

Vite serves port 5173 and proxies `/api` to the backend on port 5000.
Set `EASYSTEER_BACKEND_PORT` in both terminals when changing the backend port.
For UI-only work, choose `http://localhost:5173/mock/v1` and model
`mock-model` in Settings. This development mock returns canned text and
accepts steering specs; it does not run steering or training.

```bash
cd frontend/app
npm test
npm run typecheck
npm run build
```

The Python checks cover routing, model-cache configuration, training requests,
and payload JSON encoding. They use mocked GPU/job imports and the submodule's
CPU payload definitions:

```bash
python -m pip install -r frontend/requirements.txt numpy pytest
python -m pytest frontend/tests -q
```

## Training demonstration

`demo_training.py` trains through the Flask API. Its optional inference
step calls a separately running vllm-steer server with the saved checkpoint
converted by `easysteer.vectors.from_pyreft`:

```bash
python frontend/demo_training.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --gpu 0 --preset emoji --test \
  --api-url http://localhost:8000/v1 --api-model your-served-model
```

Run the script in the same checkout/filesystem as the job backend so it can
read `frontend/results/demo_emoji_training`. The inference server must use the
same model and declare `loreft` (or `direct` for a bias checkpoint).
`VLLM_API_KEY` supplies authentication when required. The script trains a checkpoint
and generates responses through the inference server.

Training presets use the same fields as `POST /api/train`: `model_path`,
`intervention`, `output_dir`, an array of `[input, output]` pairs in
`training_examples`, and the `reft_config` / `training_args` objects. Presets
can therefore be sent directly to the endpoint.
