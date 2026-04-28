# WasteTrade Visual Fraud Detection

Visual fraud detection MVP comparing listing photos vs load photos using DINOv2 image embeddings plus a VLM verifier (Gemini 2.5 Flash or Claude Sonnet 4.5). Built to integrate into your existing Node.js/Express + MySQL stack.

## What it does

When a seller lists waste materials with photos and later uploads container/loading photos, the system produces:

- An **aggregate similarity score** (mean of best matches per load image)
- A **min-of-max red-flag score** (the worst-matching load photo)
- A **VLM verdict** (`match` / `mismatch` / `uncertain`) with a written explanation
- A final decision: **pass** / **review** / **suspicious**

Every input, score, model version, and threshold is persisted so you can replay any check against new thresholds for free during calibration.

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│ WasteTrade frontend │         │ Reviewer dashboard  │
└──────────┬──────────┘         └──────────┬──────────┘
           │ multipart upload              │ /fraud-check/:id
           ▼                               ▼
┌──────────────────────────────────────────────────────┐
│  Node.js / Express  (this repo: node-service/)       │
│   - /api/fraud/listings                              │
│   - /api/fraud/loads                                 │
│   - /api/fraud/fraud-check  ◄── orchestrator         │
│   - cosine math in plain JS                          │
└──────┬─────────────────────────────────┬─────────────┘
       │ multipart                        │ base64
       ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────┐
│ DINOv2 sidecar       │         │ Gemini 2.5 Flash OR  │
│ (Railway, Python)    │         │ Claude Sonnet 4.5    │
│ embedding-service/   │         │ (config-flag switch) │
└──────────────────────┘         └──────────────────────┘
       │                                  │
       └──────────────┐  ┌────────────────┘
                      ▼  ▼
              ┌──────────────┐
              │ MySQL 8      │
              │ (JSON cols)  │
              └──────────────┘
```

Two reasons for the split:

1. **DINOv2 has no mature JS port.** A 50-line Python service is the right tool. Railway's free tier handles your volume with room to spare.
2. **Embedding stays warm and free.** The sidecar holds the model in memory; no per-call API charges and no cold start past first deploy.

## Why DINOv2, not CLIP

CLIP was trained on web image-caption pairs and is dominated by *scene gestalt* (warehouse, forklift, container interior). DINOv2 was trained self-supervised on raw images and captures *fine-grained texture* (bale interior, polymer transparency, label printing). For waste-bale fraud — where the warehouse always looks like a warehouse but the bale contents are what matters — DINOv2 roughly doubles fine-grained discrimination. See `docs/architecture.md` for the benchmark details.

## Why both signals, not just one

| Signal | Strength | Weakness |
|---|---|---|
| DINOv2 embedding | Calibrated continuous score; per-pair anomalies | No explanation; can't reason about polymer grade |
| VLM verdict | Human-readable reasoning; understands material taxonomy | Anchors on claimed material; verdict is discrete |

Run them in parallel on every check. The embedding gives you the threshold-tunable number; the VLM gives reviewers a reason. Combined three-way decision rule is in `node-service/src/utils/vectors.js`.

## Repository layout

```
embedding-service/        # Python FastAPI sidecar -> Railway
  app.py                  # ~200 lines, the whole service
  Dockerfile              # CPU-only, ~1.5GB final image with model baked in
  requirements.txt
  railway.toml

node-service/             # Drop into your existing Express app
  src/
    index.js              # Standalone entry (or mount into your app)
    db.js                 # mysql2 pool
    routes/fraudRoutes.js # The /api/fraud/* endpoints
    services/
      fraudCheck.js       # Orchestrator
      embeddingClient.js  # Talks to the Python sidecar
      vlmVerifier.js      # Gemini + Claude with config switch
    utils/vectors.js      # cosine, aggregation, decision rule
  migrations/
    001_init.sql          # Tables: listings, listing_images, loads, load_images, fraud_checks
    run.js
  scripts/
    smoke-test.js         # Verify everything is wired
    calibrate.js          # Tune thresholds from synthetic pairs
    replay-all.js         # Re-decide history vs new thresholds (free)
```

## Setup

### 1. Deploy the Python sidecar to Railway

```bash
cd embedding-service
# Connect this directory to a new Railway project (via the Railway CLI or web UI).
# Set environment variables in Railway:
#   EMBEDDING_API_KEY  = <generate with: openssl rand -hex 32>
#   EMBEDDING_MODEL    = facebook/dinov2-base
```

Railway will detect the Dockerfile and build automatically. Wait for the deploy and grab the public URL (something like `https://your-service.up.railway.app`). Verify:

```bash
curl https://your-service.up.railway.app/health
# {"status":"ok","model":"facebook/dinov2-base","embedding_dim":768,"device":"cpu"}
```

The first request after a cold start takes ~10s while the model loads into memory. Subsequent requests are ~200-400ms each on Railway's CPU plan.

### 2. Set up MySQL and the Node app

```bash
cd node-service
cp .env.example .env
# Edit .env with your DB credentials, EMBEDDING_SERVICE_URL, EMBEDDING_API_KEY,
# and either GEMINI_API_KEY or ANTHROPIC_API_KEY (or both).

npm install
npm run migrate         # creates the 5 tables
npm run smoke-test path/to/listing.jpg path/to/load.jpg   # verifies all wiring
npm run dev             # http://localhost:3001
```

### 3. Make your first check

```bash
# Upload listing images
curl -X POST http://localhost:3001/api/fraud/listings \
  -F "material_type=LDPE film bales" \
  -F "images=@listing1.jpg" \
  -F "images=@listing2.jpg" \
  -F "images=@listing3.jpg"
# -> {"listing_id": 1, ...}

# Upload load images
curl -X POST http://localhost:3001/api/fraud/loads \
  -F "listing_id=1" \
  -F "images=@load1.jpg" \
  -F "images=@load2.jpg" \
  -F "images=@load3.jpg"
# -> {"load_id": 1, ...}

# Run the fraud check
curl -X POST http://localhost:3001/api/fraud/fraud-check \
  -H "Content-Type: application/json" \
  -d '{"listing_id": 1, "load_id": 1}'
```

You'll get back the decision, the embedding scores, the VLM's blind description of both sets, the discrepancies it found, and the per-pair cosine matrix.

## Calibrating thresholds

The thresholds in `.env` are starting guesses. To tune them properly:

1. Build a fixtures directory:
   ```
   fixtures/
     positives/    (cases that SHOULD pass - same material in both sets)
       case_001/a/*.jpg + case_001/b/*.jpg
       ...
     negatives/    (cases that SHOULD fail - different materials)
       case_001/a/*.jpg + case_001/b/*.jpg
       ...
   ```
2. Run `npm run calibrate -- ./fixtures`
3. Read the histograms. Copy the recommended `THRESHOLD_*` values into `.env`.
4. (Optional) Run `node scripts/replay-all.js` to see how the new thresholds change historical decisions. Add `--apply` to persist.

20 positive + 20 negative cases is the bare minimum; 50+50 is meaningfully better. Bootstrap recipes for synthetic pairs are in `docs/calibration.md`.

## Switching VLMs

```bash
# In node-service/.env
VLM_PROVIDER=gemini    # default
VLM_PROVIDER=claude    # switch
```

You can also override per-request: `POST /fraud-check { listing_id, load_id, vlm_override: 'claude' }`. Both providers persist to the same `fraud_checks` row schema with `vlm_provider` and `vlm_model` columns identifying which one ran.

## Cost expectations at 5-10 loads/week

| Component | Per check | Monthly (~40 checks) |
|---|---|---|
| Railway DINOv2 sidecar | $0 (free tier) | $0 |
| Gemini 2.5 Flash (free tier on AI Studio) | $0 | $0 |
| Claude Sonnet 4.5 (if used instead) | ~$0.06 | ~$2.40 |
| MySQL (your existing) | $0 | $0 |
| **Total (Gemini)** | | **$0** |
| **Total (Claude)** | | **~$2.40** |

If you exceed Railway's free tier (you won't at this volume), the next tier is $5/month. Image storage stays on your existing server.

## Integration with the existing WasteTrade Express app

Two options:

**Option A — Standalone service** (recommended for MVP isolation):
Run `node-service/` as its own process on port 3001. Your main app calls it over HTTP. Easy to roll back, easy to monitor.

**Option B — Mount into your existing app**:
```javascript
const fraudRoutes = require('./wastetrade-fraud-detection/node-service/src/routes/fraudRoutes');
yourExistingApp.use('/api/fraud', fraudRoutes);
```
Same DB pool, same process. Less ops, more coupling.

Both work. Pick A while testing, switch to B when you're confident in the thresholds.

## What's NOT in this MVP (deliberately)

- Admin review UI — comes after threshold validation
- Dispute workflow / seller appeals
- S3 / R2 integration — your images live on the server, that's fine for now
- Vector database — at 6×6 paired comparison the JSON column is faster than any ANN index would be
- pgvector / Pinecone / Qdrant — same reason
- Auth on the routes — add this before exposing externally
- Rate limiting on the upload endpoints — same

Each of these has a clear add-on path; none is needed to validate that the comparison signal actually works on real WasteTrade loads.

## Next steps

1. Deploy the Python sidecar to Railway
2. Run migrations and smoke test on a dev MySQL
3. Run the system on 5-10 real listing/load pairs you already have
4. Build a small fixtures set and run `npm run calibrate`
5. Lock in tuned thresholds, run for 2-3 weeks, collect reviewer outcomes
6. After ~50 labeled outcomes, run `replay-all.js` to compare PR curves at different thresholds

See `docs/calibration.md` for the full tuning playbook and `docs/architecture.md` for the model/vendor decision rationale.
