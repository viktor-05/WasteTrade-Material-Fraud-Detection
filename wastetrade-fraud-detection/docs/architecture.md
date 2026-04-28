# Architecture and design decisions

This document records *why* the system is built the way it is, so that when something needs to change you can tell which decisions are load-bearing and which are arbitrary.

## Core decisions

### DINOv2 over CLIP

CLIP is the obvious default and it is the wrong default for this problem.

CLIP was trained on web image-caption pairs, which optimizes it for *semantic alignment* ("a photo of a forklift in a warehouse"). DINOv2 was trained self-supervised on raw images via a teacher-student distillation that focuses on dense visual features, which makes it strong at *fine-grained visual discrimination*.

For waste-bale fraud the warehouse, forklift, container interior, and shrink-wrap geometry are constant across legitimate and fraudulent loads. The fraud signal lives in the bale interior, polymer transparency, and label printing — exactly the textural detail DINOv2 captures and CLIP loses. On Food-101 fine-grained classification DINOv2-ViT-B/14 scores 93% versus CLIP-ViT-B/32's 65%; on iNaturalist DINOv2 hits 70% versus CLIP's 15%; on the DISC21 image-similarity benchmark DINOv2 reaches 64% top-1 versus CLIP's 28%.

Practical consequence: cosine values from DINOv2 are typically lower than from CLIP for the same image pair. Don't compare thresholds across the two. Re-calibrate from scratch if you swap models.

### CLS token, not mean-pooled patches

`outputs.last_hidden_state[:, 0, :]` is the [CLS] token. It's the canonical DINOv2 image embedding. Mean-pooling patch tokens is also valid and slightly more robust to scale variation, but the [CLS] token is what every published benchmark uses and what makes our cosine numbers comparable to anything in the literature.

### L2 normalize at the sidecar

Embeddings are normalized in `_embed_images()` before being sent back. This makes cosine similarity equal to dot product, which is faster and means every consumer of these vectors is automatically using the right metric. If you ever bypass the sidecar and embed directly, normalize before storing.

### MySQL JSON column, not a vector database

pgvector / Pinecone / Qdrant / Weaviate are designed for ANN search across millions of vectors. WasteTrade is doing a 6×6 pairwise comparison on demand. The vector database adds operational surface area — a separate service to run, a separate set of credentials to rotate, a separate index to maintain — for zero performance benefit at this scale.

When a single MySQL JSON column stops scaling: it doesn't, until you're doing similarity search across thousands of historical listings to find the closest match. We aren't doing that. We always know which listing-load pair we're comparing.

### Run embedding and VLM in parallel, not serially

The VLM call is the slowest single operation (3-8 seconds). The embedding pipeline is dominated by sidecar HTTP and DINOv2 inference (200ms × N images). Running them serially adds the VLM time on top of everything; running them in parallel hides it behind the embedding work. `Promise.all` in `fraudCheck.js` saves roughly 3-5 seconds per check.

### VLM as always-on verifier, not fallback

At 40 checks per month, even Claude Sonnet costs ~$2.40/month. There's no economic reason to gate VLM calls behind a low-similarity threshold, and there's a strong UX reason not to: reviewers need a *reason*, not a number. Always-on VLM also means every fraud_check row has both signals, which makes calibration analysis cleaner.

### Blind description before comparison

The four-step prompt in `vlmVerifier.js` forces the model to describe Set A independently, then Set B independently with an explicit "do not assume B contains the same material as A" instruction, *before* it compares them. Without this, both Gemini and Claude rationalize discrepancies away once they know what the listing claimed. With it, they catch obvious differences they would otherwise gloss over.

The `set_a_description` and `set_b_description` fields are persisted to MySQL specifically so reviewers can see whether the model actually described the load photos accurately, or hallucinated.

### Three-way decision, not binary

A two-state pass/fail forces every ambiguous case into one bucket or the other, and the cost of false negatives (missed fraud) is much higher than false positives (extra reviews). Three states let the system route ambiguity to humans:

- **pass**: embedding *and* VLM agree it's a match
- **suspicious**: either signal indicates fraud
- **review**: anything else

The matrix in `decide()` is asymmetric on purpose: either signal can trigger suspicious; only both signals together can trigger pass.

### min-of-max as a separate red flag

The aggregate (mean of best matches per load image) is robust but masks single-pallet anomalies. If 5 of 6 load photos look right but one is clearly different, the mean stays high while the minimum drops. That single low score is the smuggled-pallet pattern: legitimate cargo at the front of the container, different material at the back. We surface it as `min_of_max` and route it to suspicious independently of the aggregate.

### Persist everything, replay anything

Every fraud_check row stores the full pairwise matrix, all aggregates, the VLM's complete output, the model versions used, and the threshold values that produced the decision. This means:

1. You can re-run the decision rule against new thresholds with zero API spend (`scripts/replay-all.js`)
2. You can compare embedding-model upgrades by re-embedding only and replaying the VLM (the VLM output is preserved)
3. Every stored decision is fully auditable for disputes

The cost is ~5-10 KB per row. At 40 checks/month that's ~5MB per year. Negligible.

## Vendor choices

### Why Railway for the sidecar

Free tier covers our volume; Docker-native deployment; no cold-start surprise (the model is baked into the image at build time, not loaded at first request); HTTPS by default; easy URL to give the Node app. Fly.io and Render are both fine alternatives — the Dockerfile works on any of them. The only Railway-specific bit is `railway.toml`, which is purely configuration.

### Why Gemini AND Claude with a switch

You asked for it, but it's also the right call. Gemini Flash on the AI Studio free tier is $0 — perfect for testing and high-volume sanity runs. Claude Sonnet 4.5 has stronger material reasoning and is worth the spend on disputed cases. Having both wired up means you can A/B them on the same fixtures (`vlm_override` field), and switching providers later is one env var.

### Why no AWS Rekognition

Rekognition has face similarity and label detection. It does not have a general image-to-image embedding API. AWS's own blog redirects you to Bedrock Titan Multimodal Embeddings for this use case. Titan's $0.00006/image is cheaper than running our own sidecar in dollar terms, but: Titan auto-resizes to 384×384 internally, which loses fine texture, and we'd lose the model knowledge we get from DINOv2. Not worth the trade.

### Why no Vertex multimodalembedding@001

Same concern: Vertex auto-resizes images to 512×512 before embedding. For waste materials where bale texture is the signal, that's a disqualifier. The 1408-dim setting helps but doesn't fix the resizing.

### Why no OpenAI embeddings

OpenAI does not publish a public image embeddings API. Their cookbook tells you to run CLIP via Hugging Face. So OpenAI shows up only as a VLM option (GPT-5 nano, GPT-4o-mini), which is fine but adds a third provider for marginal gain. Skipped for MVP.

## What changes if scale goes 100x

At 1000+ loads per week:

- The Railway sidecar plan jumps to $5-20/month. Still nothing.
- VLM costs scale linearly. At 1000/week × $0.06 (Claude) = ~$240/month. Switch to Gemini Flash for everyday checks, escalate to Claude only on `uncertain` verdicts.
- MySQL JSON columns still fine. Embedding storage is ~3KB per image; even 100k images is 300MB.
- The sequential embedding loop in `_ensureEmbeddings` becomes the slowest path. Batch via the sidecar's `/embed/batch` endpoint (already implemented) and the speedup is roughly linear.
- Add a Redis cache for in-flight `(listing_id, load_id)` checks to deduplicate concurrent requests.

You don't need to change architecture. You don't need a vector DB. You don't need a queue. The pieces all scale by changing config.

## What changes if accuracy isn't good enough

If after calibration you find positive and negative score distributions overlap badly:

1. **Upgrade to dinov2-large** (1024-dim instead of 768-dim). Set `EMBEDDING_MODEL=facebook/dinov2-large` in Railway, redeploy. Bump `EMBEDDING_MODEL_VERSION` so existing cached embeddings are invalidated and recomputed lazily on next access.
2. **Center-crop before embedding**. Modify `_embed_images` in the sidecar to take a 60-70% center crop. This reduces background dominance from forklifts/walls. Cheap experiment.
3. **Switch VLM to Sonnet for everything**. Higher visual fidelity. Costs ~$2.40/month at this volume.
4. **Add SAM2 segmentation to mask non-bale regions**. Significant complexity bump but worth it if backgrounds are killing you. Don't do this without first trying 1-3.

Each step is independent. None requires schema changes or rewrites.
