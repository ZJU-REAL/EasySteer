# Web demo (frontend)

The [`frontend/`](https://github.com/ZJU-REAL/EasySteer/tree/main/frontend) module
provides a Vue interface for comparing baseline and steered outputs, chatting with
an intervention, and submitting vector extraction and ReFT training jobs.

The Flask backend runs extraction and training jobs. Generation uses a separate
[OpenAI-compatible vLLM server](openai-server.md), configured in the UI.

## Install and build

Start with a working [EasySteer installation](../getting-started/installation.md).
Use that Python environment for the backend, and install Node.js and npm for the
frontend build. From the repository root:

```bash
python -m pip install -r frontend/requirements.txt
cd frontend/app
npm ci
npm run build
cd ../..
bash frontend/start.sh
```

Open `http://localhost:8111`. The launcher serves the built UI and `/api` through
the same Flask application; backend access is also available on port 5000. It
expects dependencies and `frontend/app/dist` to be present and does not install
packages during startup. `EASYSTEER_FRONTEND_PORT` and `EASYSTEER_BACKEND_PORT`
override the respective ports.

## Start generation

In another terminal, start vLLM from the repository root:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-steer-vector --steer-algorithms all --steer-multi-vector \
  --port 8000
```

In the UI settings, set the vLLM base URL to `http://localhost:8000/v1` and the
served model to `Qwen/Qwen2.5-1.5B-Instruct`. The `all` declaration allows the
algorithms used by the demo; a deployment serving only one algorithm can declare
that algorithm explicitly instead.

URLs must be reachable from the machine running the browser. For a remote server,
forward the UI and vLLM ports or configure reachable addresses. Vector source
paths must exist on the vLLM server, and training/extraction model paths must be
accessible to the Flask backend. Training and serving load models independently,
so allocate GPU memory for the jobs you intend to run together.

## Frontend development

Run the Flask backend separately, then use the Vite development server:

```bash
# Terminal 1, from the repository root
python frontend/app.py

# Terminal 2
cd frontend/app
npm ci
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). Its
`/api` requests are proxied to the backend; generation still uses the configured
vLLM URL.

## Hosted lite demo

A lightweight [Hugging Face Space](https://huggingface.co/spaces/zjuxhl/EasySteer)
provides a smaller set of predefined interventions. See
[`hf-space/README.md`](https://github.com/ZJU-REAL/EasySteer/blob/main/hf-space/README.md)
for its API and GPU deployment modes. For extraction, training, and custom-vector
workflows, run the frontend locally.
