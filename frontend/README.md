# EasySteer Frontend

The frontend module provides a web interface with the following features:

- **Inference** — Single-vector, multi-vector, and SAE-based steering inference
- **Training** — Train steering vectors via the web UI (e.g. LoReFT)
- **Extract** — Extract steering vectors from models (e.g. DiffMean, PCA)
- **Chat** — Multi-turn chat with real-time steering interventions

## Getting Started

```bash
bash start.sh
```

The startup script will automatically:
- Check and install dependencies
- Start the backend server (default port: 5000)
- Start the frontend static server (default port: 8111)
- Open the browser automatically
