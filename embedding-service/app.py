"""
WasteTrade DINOv2 embedding sidecar.

A tiny FastAPI service whose only job is to turn an image into a 768-dim
DINOv2-base embedding (or 1024-dim DINOv2-large if you upgrade later).

Why DINOv2 and not CLIP:
  - DINOv2 was trained self-supervised on raw images and captures
    fine-grained texture and structure. CLIP was trained on web image-caption
    pairs and is dominated by 'scene gestalt' (warehouse, forklift, container)
    rather than the material inside the bale -- which is exactly the signal
    we need for waste fraud detection.
  - Empirically, DINOv2 ~doubles fine-grained discrimination on industrial
    imagery vs CLIP.

Endpoints:
  GET  /health              -> liveness probe
  POST /embed               -> {image_url} or multipart upload -> {embedding, dim}
  POST /embed/batch         -> [{image_url}, ...] -> [{embedding}, ...]
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any

import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "facebook/dinov2-base")  # 768-dim
API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # shared secret with Node app
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20.0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("embed-svc")

# ---------------------------------------------------------------------------
# Model load (once, at startup)
# ---------------------------------------------------------------------------
log.info("Loading model %s on %s ...", MODEL_NAME, DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
EMBED_DIM = model.config.hidden_size
log.info("Model loaded. embedding_dim=%d", EMBED_DIM)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WasteTrade Embedding Service",
    version="1.0.0",
    description="DINOv2 image embeddings for visual fraud detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedUrlRequest(BaseModel):
    image_url: str = Field(..., description="Public URL or signed URL to fetch.")


class EmbedBatchRequest(BaseModel):
    image_urls: list[str] = Field(..., min_length=1, max_length=32)


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int
    model: str


class EmbedBatchResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    model: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_auth(provided: str | None) -> None:
    """Reject requests without the shared secret if one is configured."""
    if not API_KEY:
        return  # unauthenticated mode (dev only)
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _load_image_from_bytes(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL throws many things
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


async def _fetch_image(url: str) -> Image.Image:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {exc}") from exc
        return _load_image_from_bytes(resp.content)


@torch.inference_mode()
def _embed_images(images: list[Image.Image]) -> np.ndarray:
    """Return L2-normalized embeddings, shape (N, EMBED_DIM)."""
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    # DINOv2 returns last_hidden_state; the [CLS] token (index 0) is the
    # global image representation. mean-pooling patch tokens also works
    # but CLS is the canonical DINOv2 image embedding.
    cls_emb = outputs.last_hidden_state[:, 0, :]  # (N, dim)
    # L2 normalize so cosine similarity = dot product later.
    cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
    return cls_emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "embedding_dim": EMBED_DIM,
        "device": DEVICE,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    image_url: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    x_api_key: str | None = Form(default=None, alias="x_api_key"),
) -> EmbedResponse:
    """Embed a single image, supplied as either a multipart file or a URL."""
    _check_auth(x_api_key)

    if image is not None:
        data = await image.read()
        img = _load_image_from_bytes(data)
    elif image_url:
        img = await _fetch_image(image_url)
    else:
        raise HTTPException(status_code=400, detail="Provide either 'image' file or 'image_url'.")

    emb = _embed_images([img])[0]
    return EmbedResponse(embedding=emb.tolist(), dim=EMBED_DIM, model=MODEL_NAME)


@app.post("/embed/json", response_model=EmbedResponse)
async def embed_json(
    payload: EmbedUrlRequest,
    x_api_key: str | None = None,
) -> EmbedResponse:
    """JSON-only variant for callers who can't easily do multipart."""
    _check_auth(x_api_key)
    img = await _fetch_image(payload.image_url)
    emb = _embed_images([img])[0]
    return EmbedResponse(embedding=emb.tolist(), dim=EMBED_DIM, model=MODEL_NAME)


@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(
    payload: EmbedBatchRequest,
    x_api_key: str | None = None,
) -> EmbedBatchResponse:
    """Embed up to 32 images by URL in one request. ~Linear speedup over single calls."""
    _check_auth(x_api_key)

    images: list[Image.Image] = []
    for url in payload.image_urls:
        images.append(await _fetch_image(url))

    embs = _embed_images(images)
    return EmbedBatchResponse(
        embeddings=[row.tolist() for row in embs],
        dim=EMBED_DIM,
        model=MODEL_NAME,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
