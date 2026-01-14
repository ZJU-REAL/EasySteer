# Stage 1: Build vllm-steer base (done separately)
# Stage 2: Add EasySteer on top of vllm-steer
FROM vllm-steer:base

LABEL maintainer="ZJU-REAL"
LABEL description="EasySteer - High-Performance LLM Steering Framework"

# Set working directory
WORKDIR /app/easysteer

# Copy EasySteer source
COPY . /app/easysteer/

# Install EasySteer dependencies
RUN pip install --no-cache-dir \
    scikit-learn \
    datasets \
    matplotlib \
    accelerate

# Install EasySteer package
RUN pip install --no-cache-dir -e .

# Environment variables for steering
ENV PYTHONUNBUFFERED=1

# Create workspace directory
RUN mkdir -p /app/workspace

# Set workspace as default
WORKDIR /app/workspace

# Reset entrypoint (vllm-openai sets ENTRYPOINT ["vllm", "serve"])
ENTRYPOINT []

# Default command
CMD ["/bin/bash"]
