# Build from hf-space/. API mode uses a remote vllm-steer server.
ARG MODE=api
# GPU mode uses an image built from this checkout with docker/build.sh.
ARG GPU_BASE=easysteer:latest
FROM python:3.12-slim AS base-api
FROM ${GPU_BASE} AS base-gpu
FROM base-${MODE} AS runtime
ARG MODE

WORKDIR /app
COPY app.py runtime.py steering_config.py algorithm_capabilities.json requirements.txt ./
COPY configs/ configs/
COPY vectors/ vectors/
COPY results/ results/
RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV DEMO_MODE=${MODE}
# Override any entrypoint inherited from the GPU base image.
ENTRYPOINT []
CMD ["python3", "app.py"]
