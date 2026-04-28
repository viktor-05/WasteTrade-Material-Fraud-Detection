#!/usr/bin/env node
/**
 * Replay every historical fraud check against the CURRENT thresholds in .env.
 *
 * Use this when you change THRESHOLD_* values: see how decision distributions
 * shift across all past checks WITHOUT spending any API money.
 *
 * USAGE:
 *   node scripts/replay-all.js [--apply]
 *
 *   --apply    Persist the new decisions to the fraud_checks table.
 *              Without this flag, it only prints what WOULD change.
 */
require('dotenv').config();
const { pool } = require('../src/db');
const { decide } = require('../src/utils/vectors');

function thresholdsFromEnv() {
  return {
    passAggregate: parseFloat(process.env.THRESHOLD_PASS_AGGREGATE || '0.60'),
    suspiciousAggregate: parseFloat(process.env.THRESHOLD_SUSPICIOUS_AGGREGATE || '0.45'),
    minOfMaxRedFlag: parseFloat(process.env.THRESHOLD_MIN_OF_MAX_RED_FLAG || '0.35'),
  };
}

async function main() {
  const apply = process.argv.includes('--apply');
  const t = thresholdsFromEnv();
  console.log('Current thresholds:', t);

  const [rows] = await pool.query(
    'SELECT id, aggregate_score, min_of_max, vlm_verdict, decision FROM fraud_checks ORDER BY id ASC'
  );
  console.log(`Replaying ${rows.length} fraud checks...\n`);

  const transitions = {};
  let changed = 0;
  for (const r of rows) {
    const result = decide({
      aggregate: parseFloat(r.aggregate_score),
      minOfMax: parseFloat(r.min_of_max),
      vlmVerdict: r.vlm_verdict,
      thresholds: t,
    });
    const before = r.decision;
    const after = result.decision;
    const key = `${before} -> ${after}`;
    transitions[key] = (transitions[key] || 0) + 1;
    if (before !== after) {
      changed += 1;
      console.log(`  fraud_check ${r.id}: ${before} -> ${after} (agg=${r.aggregate_score}, mom=${r.min_of_max}, vlm=${r.vlm_verdict})`);
      if (apply) {
        await pool.query(
          'UPDATE fraud_checks SET decision = ?, decision_reasons = ?, thresholds_used = ? WHERE id = ?',
          [after, JSON.stringify(result.reasons), JSON.stringify(t), r.id]
        );
      }
    }
  }

  console.log('\nTransition summary:');
  Object.entries(transitions).forEach(([k, v]) => console.log(`  ${k}: ${v}`));
  console.log(`\n${changed}/${rows.length} would change decision.`);
  console.log(apply ? 'Changes APPLIED.' : 'Dry run only. Pass --apply to persist.');

  await pool.end();
}

main().catch(err => {
  console.error('replay failed:', err);
  process.exit(1);
});
