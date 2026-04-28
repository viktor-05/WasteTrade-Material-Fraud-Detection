# Calibration playbook

Thresholds in `.env` are starting guesses. The real numbers come from running the system on imagery that *resembles your actual workload* and looking at the score distributions.

## When to calibrate

- Once after initial setup, before trusting any decisions
- Every time you swap the embedding model (`dinov2-base` → `dinov2-large`)
- Every time you swap the VLM provider
- After the first 50 reviewer-labeled outcomes accumulate

## Bootstrap: synthetic pairs (Day 2)

You don't have labeled fraud yet, and you can't wait for some to happen. Build synthetic positive and negative pairs from imagery you already have or can find easily.

### Synthetic positives (should pass)

For each existing legitimate listing:

1. Take its 3-6 listing photos. Split them randomly into two halves. Use one half as `a/` and the other as `b/`. This creates a "should match" pair.
2. Same trick with load photos from the same legitimate transaction.
3. Pair listing photos from one legitimate transaction with load photos from the same transaction.

Each of these should produce a `match` verdict. If they don't, your fixtures themselves are wrong (e.g., the seller actually sent something different) — investigate before trusting the calibration.

### Synthetic negatives (should fail)

Pair imagery of clearly different materials:

- LDPE film listing photos vs PET bottle load photos
- Cardboard (OCC) listing photos vs mixed plastic load photos
- Clear PET vs colored PET
- Aluminum cans vs steel cans

Public sources for waste imagery:

- **RealWaste** (Kaggle): 4,752 labeled waste images across 9 categories. Best single source.
- **TrashNet**: classic 6-category dataset, smaller but clean
- **Garbage Classification** (Kaggle): 12 categories
- Targeted Google image search: `"LDPE bale" site:alibaba.com`, `"PET bale" site:made-in-china.com`, etc.

You want each negative case to have 3-6 images on each side. Don't use single-image negatives — that doesn't exercise the aggregation logic.

### Directory layout

```
fixtures/
  positives/
    case_001/
      a/
        listing_1.jpg
        listing_2.jpg
        listing_3.jpg
      b/
        load_1.jpg
        load_2.jpg
        load_3.jpg
    case_002/
      ...
  negatives/
    case_001/
      a/
        ldpe_listing_1.jpg
        ldpe_listing_2.jpg
        ldpe_listing_3.jpg
      b/
        pet_load_1.jpg
        pet_load_2.jpg
        pet_load_3.jpg
    ...
```

Minimum: 20 positive + 20 negative cases. Recommended: 50 + 50.

### Run it

```bash
cd node-service
npm run calibrate -- ./fixtures
```

The script embeds every image, computes scores for every case, prints distribution statistics and ASCII histograms, and recommends thresholds.

## Reading the histograms

Healthy outcome:

```
positives aggregate (n=50)
  0.40-0.45 |
  0.45-0.50 |
  0.50-0.55 |
  0.55-0.60 | ###
  0.60-0.65 | #########
  0.65-0.70 | ################
  0.70-0.75 | ###########
  0.75-0.80 | ####
  0.80-0.85 |

negatives aggregate (n=50)
  0.20-0.25 | ##
  0.25-0.30 | #######
  0.30-0.35 | ############
  0.35-0.40 | ##########
  0.40-0.45 | ####
  0.45-0.50 | ##
  0.50-0.55 |
```

Two clearly separated humps with a gap between them. Pick `THRESHOLD_PASS_AGGREGATE` near the bottom of the positive hump (P10) and `THRESHOLD_SUSPICIOUS_AGGREGATE` near the top of the negative hump (P90). The gap becomes the "review" zone.

Unhealthy outcome:

```
positives aggregate (n=20)
  0.40-0.45 | ##
  0.45-0.50 | ###
  0.50-0.55 | ####
  0.55-0.60 | #####
  0.60-0.65 | ####
  0.65-0.70 | ##

negatives aggregate (n=20)
  0.40-0.45 | ###
  0.45-0.50 | #####
  0.50-0.55 | ####
  0.55-0.60 | ###
  0.60-0.65 | ##
  0.65-0.70 | #
```

Heavy overlap. Either:

1. Your fixtures are wrong (negatives accidentally contain similar materials, or positives genuinely look different from each other due to e.g. different lighting). Audit them.
2. dinov2-base lacks discrimination on your specific materials. Upgrade to dinov2-large.
3. Background is dominating the embedding (warehouse vs container shows up as similarity even though materials differ). Add a center-crop preprocessing step.

## Production: real labeled outcomes

Once you've been running for a few weeks and reviewers have submitted outcomes via `POST /fraud-check/:id/outcome`, you have proper labels.

### Pulling the data

```sql
SELECT
  aggregate_score,
  min_of_max,
  vlm_verdict,
  decision,
  reviewer_outcome,
  created_at
FROM fraud_checks
WHERE reviewer_outcome IS NOT NULL
ORDER BY id ASC;
```

### Metrics that matter (and ones that don't)

**Don't use:**
- **Accuracy** — Fraud is rare; predicting "pass" for everything will give you 95% accuracy and miss every fraud.
- **ROC-AUC** — Misleading on imbalanced data, masks the false-positive rate at the operating point you actually use.

**Do use:**
- **PR-AUC** (precision-recall area under curve) — Honest on imbalanced data
- **F2 score** — Weights recall 4x more than precision because missing fraud costs more than re-reviewing a legitimate load
- **MCC** (Matthews Correlation Coefficient) — Single number that's robust to imbalance

### Threshold sweep

After ~50 labeled outcomes:

1. Treat `confirmed_fraud` as the positive class
2. For each candidate threshold τ in {0.3, 0.35, ..., 0.8}:
   - Compute the decision the system would have made (`suspicious` if `aggregate ≤ τ` etc.)
   - Compute precision, recall, F2 vs reviewer labels
3. Plot precision vs recall
4. Pick the operating point matching your business cost ratio. For waste-trade fraud, recall over precision is the right call: **F2 around 0.7-0.8** is a reasonable target.

The replay script makes this analysis cheap: edit `.env`, run `node scripts/replay-all.js` (no `--apply`), see the transition counts. Iterate until you like the numbers, then run with `--apply`.

## Re-calibration triggers

Re-run calibration when:

- The mix of materials on the platform shifts significantly
- Sellers start uploading meaningfully different photo types (e.g., from camera-phone snapshots to professional product photography)
- A new fraud pattern emerges that the current thresholds miss
- Reviewer disagreement with system decisions exceeds ~15%

Don't re-calibrate on every fraud case. Single-case adjustments lead to thrashing. Wait for trends across at least 10 labeled outcomes before changing thresholds.

## Failure modes specific to waste imagery

Watch for these during calibration; they're the most common reasons positive and negative distributions overlap:

| Failure | Symptom | Mitigation |
|---|---|---|
| Lighting variation (warehouse fluorescent vs container daylight) | Same-material pairs score lower than expected | Train reviewers to weight VLM verdict more heavily; VLM handles this better than embeddings |
| Indoor vs container backgrounds | Different-material pairs score *higher* than expected because warehouse looks like warehouse | Center-crop before embedding; relies on min-of-max as second signal |
| Polymer-grade discrimination (LDPE 99/1 vs 95/5) | Both signals score high regardless of grade | Genuinely hard problem; route all such cases to review by default |
| Scale mismatch (close-up texture vs wide forklift shot) | Pairwise cosines are bimodal — some very high, some very low | Aggregation handles this; require sellers to upload at least one wide and one close-up per side |
| Background clutter (forklifts, pallets, shrink-wrap) | Embedding scores look stuck around 0.55-0.65 regardless of material | Most fixable with center crop; dinov2-large helps too |

Build a **regression set** as you encounter these in production: 1-2 examples of each scenario, labeled, kept separately from training fixtures, re-run after every model or threshold change. This single QA practice catches more regressions than any benchmark.
