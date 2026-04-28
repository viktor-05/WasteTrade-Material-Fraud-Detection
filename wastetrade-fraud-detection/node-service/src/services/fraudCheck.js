/**
 * Fraud check orchestrator.
 *
 * The seven-step pipeline:
 *   1. Fetch image rows from MySQL for the given listing + load
 *   2. For any image without a cached embedding, embed it via the sidecar
 *      and persist the vector to MySQL JSON column
 *   3. Build the pairwise cosine matrix in JS
 *   4. Aggregate to (aggregate, minOfMax) plus per-load-image best matches
 *   5. Call the VLM verifier in parallel with the embedding work
 *   6. Apply the decision rule
 *   7. Persist a fraud_checks row that captures EVERYTHING for replay
 */
const { pool } = require('../db');
const embeddingClient = require('./embeddingClient');
const vlmVerifier = require('./vlmVerifier');
const { pairwiseCosine, aggregateScores, decide } = require('../utils/vectors');

function _thresholds() {
  return {
    passAggregate: parseFloat(process.env.THRESHOLD_PASS_AGGREGATE || '0.60'),
    suspiciousAggregate: parseFloat(process.env.THRESHOLD_SUSPICIOUS_AGGREGATE || '0.45'),
    minOfMaxRedFlag: parseFloat(process.env.THRESHOLD_MIN_OF_MAX_RED_FLAG || '0.35'),
  };
}

/**
 * Fetch images for a listing or load. Returns rows with {id, path, embedding, embedding_model}.
 * `kind` is 'listing' or 'load'.
 */
async function _fetchImages(kind, parentId) {
  const table = kind === 'listing' ? 'listing_images' : 'load_images';
  const fk = kind === 'listing' ? 'listing_id' : 'load_id';
  const [rows] = await pool.query(
    `SELECT id, path, embedding, embedding_model FROM ${table} WHERE ${fk} = ? ORDER BY id ASC`,
    [parentId]
  );
  return rows;
}

/**
 * Ensure every image row has an embedding for the current model version.
 * Writes any newly computed embeddings back to MySQL. Returns the in-memory
 * array of {id, path, embedding} ready for the math step.
 */
async function _ensureEmbeddings(kind, rows) {
  const table = kind === 'listing' ? 'listing_images' : 'load_images';
  const expectedModel = embeddingClient.MODEL_VERSION;

  const result = [];
  for (const row of rows) {
    const cached = row.embedding && row.embedding_model === expectedModel ? row.embedding : null;
    if (cached) {
      // mysql2 may return the JSON column as a string OR already-parsed array
      // depending on driver/mysql version. Handle both.
      const vector = Array.isArray(cached) ? cached : JSON.parse(cached);
      result.push({ id: row.id, path: row.path, embedding: vector });
      continue;
    }

    // Compute embedding via the sidecar.
    const { embedding } = await embeddingClient.embedFile(row.path);
    await pool.query(
      `UPDATE ${table} SET embedding = ?, embedding_model = ? WHERE id = ?`,
      [JSON.stringify(embedding), expectedModel, row.id]
    );
    result.push({ id: row.id, path: row.path, embedding });
  }
  return result;
}

/**
 * Run the full fraud check.
 *
 * @param {object} params
 * @param {number} params.listingId
 * @param {number} params.loadId
 * @param {string} [params.claimedMaterial]
 * @param {string} [params.vlmOverride] - optional 'gemini' | 'claude'
 */
async function runFraudCheck({ listingId, loadId, claimedMaterial, vlmOverride }) {
  const startedAt = Date.now();

  // 1. Fetch image rows.
  const [listingRowsRaw, loadRowsRaw] = await Promise.all([
    _fetchImages('listing', listingId),
    _fetchImages('load', loadId),
  ]);

  if (!listingRowsRaw.length) throw new Error(`No listing images found for listing_id=${listingId}`);
  if (!loadRowsRaw.length) throw new Error(`No load images found for load_id=${loadId}`);

  // 2 + 5. Run embedding work and VLM in parallel. The VLM call is the
  // slowest single op (3-8s) so overlapping it with the embedding pipeline
  // saves real wall time.
  const [listingImgs, loadImgs, vlmResult] = await Promise.all([
    _ensureEmbeddings('listing', listingRowsRaw),
    _ensureEmbeddings('load', loadRowsRaw),
    vlmVerifier.verify({
      listingImagePaths: listingRowsRaw.map(r => r.path),
      loadImagePaths: loadRowsRaw.map(r => r.path),
      claimedMaterial,
      providerOverride: vlmOverride,
    }).catch(err => {
      // Don't fail the whole check if the VLM call breaks - record it and
      // fall back to embedding-only decision-making.
      console.error('[fraud-check] VLM verifier failed:', err.message);
      return { provider: 'error', model: 'error', verdict: 'uncertain', confidence: 0, discrepancies: [], summary: `VLM error: ${err.message}` };
    }),
  ]);

  // 3 + 4. Pairwise cosine -> aggregates.
  const matrix = pairwiseCosine(
    listingImgs.map(r => r.embedding),
    loadImgs.map(r => r.embedding)
  );
  const { aggregate, minOfMax, maxPerLoadImage } = aggregateScores(matrix);

  // Build a richer per-pair payload for the review screen.
  const pairwiseDetail = [];
  for (let i = 0; i < listingImgs.length; i++) {
    for (let j = 0; j < loadImgs.length; j++) {
      pairwiseDetail.push({
        listing_image_id: listingImgs[i].id,
        load_image_id: loadImgs[j].id,
        cosine: matrix[i][j],
      });
    }
  }

  // 6. Decision rule.
  const thresholds = _thresholds();
  const { decision, reasons } = decide({
    aggregate,
    minOfMax,
    vlmVerdict: vlmResult.verdict,
    thresholds,
  });

  const elapsedMs = Date.now() - startedAt;

  // 7. Persist the full record. Storing the raw matrix + thresholds used +
  // model versions makes it possible to replay this exact check against
  // updated thresholds later, which is essential during calibration.
  const checkRecord = {
    listing_id: listingId,
    load_id: loadId,
    claimed_material: claimedMaterial || null,
    pairwise_scores: JSON.stringify(pairwiseDetail),
    max_per_load_image: JSON.stringify(maxPerLoadImage),
    aggregate_score: aggregate,
    min_of_max: minOfMax,
    embedding_model: embeddingClient.MODEL_VERSION,
    vlm_provider: vlmResult.provider,
    vlm_model: vlmResult.model,
    vlm_verdict: vlmResult.verdict,
    vlm_confidence: vlmResult.confidence,
    vlm_discrepancies: JSON.stringify(vlmResult.discrepancies || []),
    vlm_summary: vlmResult.summary || null,
    vlm_set_a_description: vlmResult.set_a_description || null,
    vlm_set_b_description: vlmResult.set_b_description || null,
    vlm_comparison: vlmResult.comparison || null,
    decision,
    decision_reasons: JSON.stringify(reasons),
    thresholds_used: JSON.stringify(thresholds),
    elapsed_ms: elapsedMs,
  };

  const [insertResult] = await pool.query('INSERT INTO fraud_checks SET ?', checkRecord);
  const fraudCheckId = insertResult.insertId;

  return {
    fraud_check_id: fraudCheckId,
    listing_id: listingId,
    load_id: loadId,
    decision,
    decision_reasons: reasons,
    scores: {
      aggregate,
      min_of_max: minOfMax,
      max_per_load_image: maxPerLoadImage,
    },
    vlm: {
      provider: vlmResult.provider,
      model: vlmResult.model,
      verdict: vlmResult.verdict,
      confidence: vlmResult.confidence,
      discrepancies: vlmResult.discrepancies,
      summary: vlmResult.summary,
      set_a_description: vlmResult.set_a_description,
      set_b_description: vlmResult.set_b_description,
      comparison: vlmResult.comparison,
    },
    pairwise_scores: pairwiseDetail,
    embedding_model: embeddingClient.MODEL_VERSION,
    thresholds_used: thresholds,
    elapsed_ms: elapsedMs,
  };
}

/**
 * Replay a stored fraud check against current thresholds. Used during
 * calibration: change THRESHOLD_* in .env, run replay over historical
 * checks, see how decision distributions shift WITHOUT spending any API
 * money on re-embedding or re-verifying.
 */
async function replayFraudCheck(fraudCheckId) {
  const [rows] = await pool.query('SELECT * FROM fraud_checks WHERE id = ?', [fraudCheckId]);
  if (!rows.length) throw new Error(`No fraud_check with id=${fraudCheckId}`);
  const record = rows[0];

  const thresholds = _thresholds();
  const { decision, reasons } = decide({
    aggregate: record.aggregate_score,
    minOfMax: record.min_of_max,
    vlmVerdict: record.vlm_verdict,
    thresholds,
  });

  return {
    fraud_check_id: fraudCheckId,
    original_decision: record.decision,
    new_decision: decision,
    new_reasons: reasons,
    thresholds_used: thresholds,
    aggregate_score: record.aggregate_score,
    min_of_max: record.min_of_max,
    vlm_verdict: record.vlm_verdict,
  };
}

module.exports = {
  runFraudCheck,
  replayFraudCheck,
};
