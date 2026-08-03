# Web demo (frontend)

The [`frontend/`](https://github.com/ZJU-REAL/EasySteer/tree/main/frontend) module
provides a web interface for using EasySteer without writing code: configure models,
adjust steering parameters, test both steering and ReFT interventions, compare baseline
outputs with steered results side by side, and chat multi-turn with a steered model.

## Run locally (full-featured)

Requires a working [EasySteer installation](../getting-started/installation.md) and a
GPU:

```bash
cd frontend
bash start.sh
```

## Hosted lite demo

A lightweight demo runs on Hugging Face Spaces for quick testing without any setup:
[huggingface.co/spaces/zjuxhl/EasySteer](https://huggingface.co/spaces/zjuxhl/EasySteer)
(the [`hf-space/`](https://github.com/ZJU-REAL/EasySteer/tree/main/hf-space) directory
holds its source). For the full feature set, run the frontend locally.

<!-- TODO: screenshots, a tour of the tabs (vector testing / training / chat), and the
frontend's server/port configuration. -->
