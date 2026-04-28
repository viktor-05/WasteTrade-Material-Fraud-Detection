# syntax=docker/dockerfile:1.7
# CPU-only image for Railway. ~1.5GB final size with model cached.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

# System libs needed by Pillow + torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev zlib1g-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the model into the image so first request isn't a 30s cold start.
# This adds ~400MB to the image but eliminates Railway warmup pain.
ARG EMBEDDING_MODEL=facebook/dinov2-base
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('${EMBEDDING_MODEL}'); \
    AutoModel.from_pretrained('${EMBEDDING_MODEL}')"

COPY app.py .

# Railway sets $PORT. Default to 8000 for local dev.
ENV PORT=8000
EXPOSE 8000

# Single worker is correct: model is in-memory and CPU-bound; multiple workers
# would multiply RAM by N for no throughput gain at this volume.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
