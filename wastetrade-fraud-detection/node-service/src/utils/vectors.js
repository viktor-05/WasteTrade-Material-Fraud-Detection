/**
 * Vector math for fraud detection.
 *
 * All embeddings from the sidecar are L2-normalized, so cosine similarity
 * is just a dot product. We keep the math here in plain JS - at this volume
 * (< 50 image comparisons per check) there's no point pulling in a numerical
 * library.
 */

/**
 * Cosine similarity between two L2-normalized vectors.
 * If you ever feed in unnormalized vectors, this will silently return wrong
 * numbers. The sidecar always normalizes - keep it that way.
 */
function cosineSim(a, b) {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  // Numerical safety: cosine of L2-normalized vectors is in [-1, 1] but
  // floating point can produce 1.0000000002. Clamp.
  return Math.max(-1, Math.min(1, dot));
}

/**
 * Compute the full pairwise cosine matrix between two sets of embeddings.
 * Returns a 2D array of shape [listingEmbeds.length][loadEmbeds.length].
 */
function pairwiseCosine(listingEmbeds, loadEmbeds) {
  const matrix = [];
  for (let i = 0; i < listingEmbeds.length; i++) {
    const row = [];
    for (let j = 0; j < loadEmbeds.length; j++) {
      row.push(cosineSim(listingEmbeds[i], loadEmbeds[j]));
    }
    matrix.push(row);
  }
  return matrix;
}

/**
 * Aggregate the pairwise matrix into the two scores that drive decisions.
 *
 * Why these specific aggregations:
 *   - For each LOAD image, find its best match among LISTING images.
 *     A scale-mismatched close-up should still find SOME wide listing
 *     photo it matches well, so taking max-per-load is robust to the
 *     mix of angles/zooms the seller uploaded.
 *   - aggregate = mean of those maxes -> the headline "do these belong
 *     together" score.
 *   - minOfMax = the worst load image's best match -> a single low number
 *     here is a much stronger fraud signal than a low average. If 5 of
 *     6 load photos look right but one is clearly different, that's
 *     exactly the smuggled-pallet pattern.
 */
function aggregateScores(matrix) {
  if (matrix.length === 0 || matrix[0].length === 0) {
    return { aggregate: 0, minOfMax: 0, maxPerLoadImage: [] };
  }

  const numListing = matrix.length;
  const numLoad = matrix[0].length;

  // For each load image (column), find max over all listing images (rows).
  const maxPerLoadImage = [];
  for (let j = 0; j < numLoad; j++) {
    let best = -Infinity;
    for (let i = 0; i < numListing; i++) {
      if (matrix[i][j] > best) best = matrix[i][j];
    }
    maxPerLoadImage.push(best);
  }

  const aggregate = maxPerLoadImage.reduce((s, x) => s + x, 0) / maxPerLoadImage.length;
  const minOfMax = Math.min(...maxPerLoadImage);

  return { aggregate, minOfMax, maxPerLoadImage };
}

/**
 * Apply the three-way decision rule combining the embedding signals and
 * the VLM verdict.
 *
 * The matrix below intentionally NEVER lets either signal alone produce a
 * 'pass'. A false 'pass' on actual fraud is much more expensive than a
 * false 'review'. The rules:
 *
 *   - PASS:        embedding strong AND VLM says match
 *   - SUSPICIOUS:  embedding very weak OR minOfMax red flag OR VLM says mismatch
 *   - REVIEW:      anything else (the entire ambiguous middle)
 */
function decide({ aggregate, minOfMax, vlmVerdict, thresholds }) {
  const {
    passAggregate,
    suspiciousAggregate,
    minOfMaxRedFlag,
  } = thresholds;

  const reasons = [];

  // Hard suspicious triggers.
  if (vlmVerdict === 'mismatch') {
    reasons.push('VLM verdict: mismatch');
    return { decision: 'suspicious', reasons };
  }
  if (aggregate <= suspiciousAggregate) {
    reasons.push(`aggregate ${aggregate.toFixed(3)} <= suspicious threshold ${suspiciousAggregate}`);
    return { decision: 'suspicious', reasons };
  }
  if (minOfMax <= minOfMaxRedFlag) {
    reasons.push(`min-of-max ${minOfMax.toFixed(3)} <= red flag threshold ${minOfMaxRedFlag} (one load image looks very different)`);
    return { decision: 'suspicious', reasons };
  }

  // Pass requires BOTH signals to be positive.
  if (aggregate >= passAggregate && vlmVerdict === 'match') {
    reasons.push(`aggregate ${aggregate.toFixed(3)} >= pass threshold ${passAggregate} and VLM says match`);
    return { decision: 'pass', reasons };
  }

  // Everything else: human review.
  reasons.push(
    `aggregate ${aggregate.toFixed(3)} in ambiguous range or VLM verdict '${vlmVerdict}' uncertain`
  );
  return { decision: 'review', reasons };
}

module.exports = {
  cosineSim,
  pairwiseCosine,
  aggregateScores,
  decide,
};
