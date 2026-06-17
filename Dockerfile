# ============================================================================
# Dockerfile for the FastAPI backend (inference service).
#
# Default base is CPU-friendly so the image builds/runs anywhere. For GPU
# serving, switch the base image to an NVIDIA CUDA runtime (see comment below)
# and run with `--gpus all`.
# ============================================================================
FROM python:3.11-slim AS base

# --- For GPU serving, replace the FROM above with: ---------------------------
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# and install python3.11 + pip in this layer.
# -----------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# System deps (git for HF downloads, build tools for some wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to leverage Docker layer caching.
# We use the lean SERVING requirements (CPU torch, no bitsandbytes/trl) so the
# image builds on Apple Silicon (arm64) and stays small.
COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

# Copy only what the serving path needs.
COPY src/ ./src/
COPY api/ ./api/
COPY pyproject.toml .

# Adapter weights are mounted as a volume at runtime (see docker-compose),
# keeping the image small and the model swappable without a rebuild.
ENV ADAPTER_PATH=/app/outputs/healthcare-qlora-adapter

EXPOSE 8000

# Container healthcheck used by docker-compose `depends_on: condition`.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
