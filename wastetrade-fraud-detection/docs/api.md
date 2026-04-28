# API reference

All endpoints are under `/api/fraud/` when mounted into your Express app, or `/api/fraud/` standalone on port 3001.

## Listings

### `POST /api/fraud/listings`

Create a listing and upload its photos.

**Request** (`multipart/form-data`):
- `material_type` (string, required): e.g. `"LDPE film bales"`
- `images` (file, 1-10): the listing photos

**Response** `201`:
```json
{
  "listing_id": 42,
  "material_type": "LDPE film bales",
  "images_uploaded": 3
}
```

### `POST /api/fraud/listings/:id/images`

Add more photos to an existing listing.

**Request** (`multipart/form-data`): `images` (1-10 files)

### `GET /api/fraud/listings/:id`

```json
{
  "id": 42,
  "material_type": "LDPE film bales",
  "created_at": "2026-04-28T10:30:00Z",
  "images": [
    { "id": 101, "filename": "...", "path": "...", "embedding_model": null, "created_at": "..." }
  ]
}
```

`embedding_model` is `null` until the first fraud check triggers embedding for that image.

## Loads

### `POST /api/fraud/loads`

Create a load tied to a listing and upload its photos.

**Request** (`multipart/form-data`):
- `listing_id` (integer, required)
- `images` (file, 1-10): the load/container photos

**Response** `201`: `{ "load_id": 99, "listing_id": 42, "images_uploaded": 4 }`

### `GET /api/fraud/loads/:id`

Same shape as the listing endpoint.

## Fraud check

### `POST /api/fraud/fraud-check`

Run a fraud check between a listing and a load.

**Request** (`application/json`):
```json
{
  "listing_id": 42,
  "load_id": 99,
  "claimed_material": "LDPE film bales (optional, defaults to listing.material_type)",
  "vlm_override": "claude (optional, overrides VLM_PROVIDER env)"
}
```

**Response** `200`:
```json
{
  "fraud_check_id": 1,
  "listing_id": 42,
  "load_id": 99,
  "decision": "review",
  "decision_reasons": [
    "aggregate 0.532 in ambiguous range or VLM verdict 'uncertain' uncertain"
  ],
  "scores": {
    "aggregate": 0.5321,
    "min_of_max": 0.4102,
    "max_per_load_image": [0.6711, 0.5832, 0.4102, 0.5640]
  },
  "vlm": {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "verdict": "uncertain",
    "confidence": 0.62,
    "discrepancies": [
      "Listing photos show predominantly white film with blue strapping; load photo 3 shows mixed-color film fragments not visible in any listing photo",
      "Bale density appears lower in load photos than in listing photos"
    ],
    "summary": "Same broad material family (LDPE film bales) but visible color and density differences in load photo 3 that warrant manual inspection.",
    "set_a_description": "Three white LDPE film bales bound with blue plastic straps...",
    "set_b_description": "Four bales in a container; first three appear similar to set A but the third...",
    "comparison": "Set A is uniformly white film bales..."
  },
  "pairwise_scores": [
    { "listing_image_id": 101, "load_image_id": 201, "cosine": 0.6711 },
    { "listing_image_id": 101, "load_image_id": 202, "cosine": 0.5832 }
  ],
  "embedding_model": "dinov2-base-v1",
  "thresholds_used": { "passAggregate": 0.6, "suspiciousAggregate": 0.45, "minOfMaxRedFlag": 0.35 },
  "elapsed_ms": 4127
}
```

Decisions are one of `pass`, `review`, `suspicious`. See [architecture.md](./architecture.md) for the rule.

### `GET /api/fraud/fraud-check/:id`

Returns the full stored row for a fraud check.

### `POST /api/fraud/fraud-check/:id/replay`

Re-decides this check against the *current* `THRESHOLD_*` values in `.env` without re-spending any API money.

**Response**:
```json
{
  "fraud_check_id": 1,
  "original_decision": "review",
  "new_decision": "suspicious",
  "new_reasons": ["aggregate 0.532 <= suspicious threshold 0.55"],
  "thresholds_used": { "passAggregate": 0.65, "suspiciousAggregate": 0.55, "minOfMaxRedFlag": 0.4 },
  "aggregate_score": 0.5321,
  "min_of_max": 0.4102,
  "vlm_verdict": "uncertain"
}
```

The decision in the database is NOT updated by this call. Use `scripts/replay-all.js --apply` if you want to persist new decisions across all historical checks.

### `POST /api/fraud/fraud-check/:id/outcome`

Reviewer feedback. This is what calibration runs against.

**Request**:
```json
{
  "outcome": "confirmed_fraud" | "false_alarm" | "inconclusive",
  "reviewer_notes": "string (optional)"
}
```

### `GET /api/fraud/fraud-check`

List checks (paginated).

**Query params**:
- `limit` (default 50, max 200)
- `offset` (default 0)
- `decision` (optional filter: `pass` | `review` | `suspicious`)

## Health

### `GET /health`

Returns the overall service status, MySQL reachability, and the embedding sidecar's reported state. Used by Railway/uptime monitors.

```json
{
  "status": "ok",
  "service": "wastetrade-fraud-detection",
  "db": "ok",
  "embedding_sidecar": {
    "status": "ok",
    "model": "facebook/dinov2-base",
    "embedding_dim": 768,
    "device": "cpu"
  }
}
```

Returns `503` if any component is degraded.

## Embedding sidecar (Python service - usually called only by the Node app)

### `GET /health`
### `POST /embed` (multipart: `image` file or `image_url` field, optional `x_api_key`)
### `POST /embed/json` ({ image_url })
### `POST /embed/batch` ({ image_urls: [...] })

All return `{ embedding: [...], dim, model }` with L2-normalized 768-dim (base) or 1024-dim (large) vectors.

The Node `embeddingClient` handles all of this for you; you should rarely call the sidecar directly.
