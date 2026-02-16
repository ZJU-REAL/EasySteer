# ============================================================
# EasySteer HF Space Dockerfile
#
# Switch mode by changing the MODE build-arg (default: api):
#
#   API mode (lightweight ~200 MB, CPU, calls remote vLLM server):
#     docker build --build-arg MODE=api -t easysteer-demo .
#
#   GPU mode (full vLLM + model weights, requires NVIDIA GPU):
#     docker build --build-arg MODE=gpu -t easysteer-demo .
# ============================================================

ARG MODE=api

# ---- Stage aliases for base images ----
FROM python:3.10-slim                     AS base-api
FROM xuhaolei/easysteer:easysteer-demo    AS base-gpu

# ---- Runtime (selected by MODE) ----
FROM base-${MODE} AS runtime

WORKDIR /app

# Copy application source & configs (always needed)
COPY app.py .
COPY requirements.txt .
COPY configs/ configs/

# vectors/ and results/ are used by GPU mode; harmless in API mode.
# Comment out the next two lines if the directories don't exist.
COPY vectors/ vectors/
COPY results/ results/

# Install Python deps
# (GPU image already has torch / vLLM; reinstalling gradio + openai is fast)
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 7860

# ---- Environment defaults ----
# DEMO_MODE: "api" | "gpu" | "auto" (auto = detect from VLLM_API_URL)
# Override in HF Space → Settings → Variables / Secrets as needed.
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV DEMO_MODE="auto"

CMD ["python3", "app.py"]
