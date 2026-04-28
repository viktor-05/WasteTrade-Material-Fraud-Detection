#!/usr/bin/env node
/**
 * Threshold calibration script.
 *
 * USAGE:
 *   node scripts/calibrate.js <fixtures_dir>
 *
 * EXPECTED LAYOUT:
 *   fixtures_dir/
 *     positives/
 *       case_001/
 *         a/  (3+ images that go together - e.g. listing photos)
 *         b/  (3+ images that ALSO go together - e.g. load photos of same material)
 *       case_002/
 *         a/ ...
 *         b/ ...
 *     negatives/
 *       case_001/
 *         a/  (e.g. LDPE listing photos)
 *         b/  (e.g. PET load photos - clearly different material)
 *       ...
 *
 * WHAT IT DOES:
 *   1. For each case, embeds every image via the sidecar.
 *   2. Computes the aggregate + minOfMax scores - the same numbers the
 *      production fraud_check would compute.
 *   3. Plots histograms of positive vs negative score distributions in the
 *      terminal (text-based, no plotting libs).
 *   4. Recommends thresholds at the right percentiles of each distribution.
 *
 * WHY:
 *   You can't reasonably set THRESHOLD_PASS_AGGREGATE / THRESHOLD_SUSPICIOUS_AGGREGATE
 *   from a blog post - DINOv2 cosine values vary with your specific imagery
 *   (lighting, materials, bale formats). This script gives you values
 *   grounded in YOUR fixtures.
 *
 *   Bootstrap recommendation: 20 positive cases + 20 negative cases is the
 *   minimum to get useful percentile estimates. 50+50 is meaningfully better.
 */
require('dotenv').config();
const fs = require('fs');
const path = require('path');

const embeddingClient = require('../src/services/embeddingClient');
const { pairwiseCosine, aggregateScores } = require('../src/utils/vectors');

function listImagesIn(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(f => /\.(jpg|jpeg|png|webp)$/i.test(f))
    .map(f => path.join(dir, f))
    .sort();
}

function listCases(parent) {
  if (!fs.existsSync(parent)) return [];
  return fs.readdirSync(parent)
    .map(name => {
      const dir = path.join(parent, name);
      if (!fs.statSync(dir).isDirectory()) return null;
      const a = listImagesIn(path.join(dir, 'a'));
      const b = listImagesIn(path.join(dir, 'b'));
      if (!a.length || !b.length) {
        console.warn(`  skipping ${name}: needs both a/ and b/ subfolders with images`);
        return null;
      }
      return { name, a, b };
    })
    .filter(Boolean);
}

async function embedAll(paths, label) {
  const embeddings = [];
  for (const p of paths) {
    process.stdout.write(`    [${label}] embedding ${path.basename(p)}... `);
    const t = Date.now();
    const { embedding } = await embeddingClient.embedFile(p);
    embeddings.push(embedding);
    process.stdout.write(`${Date.now() - t}ms\n`);
  }
  return embeddings;
}

async function scoreCase({ a, b }) {
  const [aEmb, bEmb] = await Promise.all([
    embedAll(a, 'A'),
    embedAll(b, 'B'),
  ]);
  const matrix = pairwiseCosine(aEmb, bEmb);
  return aggregateScores(matrix);
}

function percentile(arr, p) {
  if (!arr.length) return NaN;
  const sorted = [...arr].sort((x, y) => x - y);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
  return sorted[idx];
}

function stats(arr) {
  if (!arr.length) return { n: 0 };
  const sorted = [...arr].sort((a, b) => a - b);
  return {
    n: arr.length,
    min: sorted[0],
    p10: percentile(arr, 10),
    p25: percentile(arr, 25),
    median: percentile(arr, 50),
    p75: percentile(arr, 75),
    p90: percentile(arr, 90),
    max: sorted[sorted.length - 1],
    mean: arr.reduce((s, x) => s + x, 0) / arr.length,
  };
}

function asciiHistogram(values, label, bins = 20, width = 40) {
  const min = 0;
  const max = 1;
  const bucket = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / bucket)));
    counts[idx] += 1;
  });
  const peak = Math.max(...counts, 1);
  console.log(`\n  ${label} (n=${values.length})`);
  for (let i = 0; i < bins; i++) {
    const lo = (min + i * bucket).toFixed(2);
    const hi = (min + (i + 1) * bucket).toFixed(2);
    const bar = '#'.repeat(Math.round((counts[i] / peak) * width));
    console.log(`  ${lo}-${hi} | ${bar} ${counts[i]}`);
  }
}

async function main() {
  const fixturesDir = process.argv[2];
  if (!fixturesDir) {
    console.error('Usage: node scripts/calibrate.js <fixtures_dir>');
    process.exit(1);
  }

  const positives = listCases(path.join(fixturesDir, 'positives'));
  const negatives = listCases(path.join(fixturesDir, 'negatives'));

  console.log(`Found ${positives.length} positive cases and ${negatives.length} negative cases.`);
  if (positives.length < 5 || negatives.length < 5) {
    console.warn('WARNING: fewer than 5 cases per class. Recommended: 20+20 minimum, 50+50 preferred.');
  }

  const posAgg = [];
  const posMinMax = [];
  const negAgg = [];
  const negMinMax = [];

  console.log('\n--- POSITIVES (should pass) ---');
  for (const c of positives) {
    console.log(`\n  case: ${c.name}`);
    const s = await scoreCase(c);
    console.log(`    aggregate=${s.aggregate.toFixed(4)} minOfMax=${s.minOfMax.toFixed(4)}`);
    posAgg.push(s.aggregate);
    posMinMax.push(s.minOfMax);
  }

  console.log('\n--- NEGATIVES (should fail) ---');
  for (const c of negatives) {
    console.log(`\n  case: ${c.name}`);
    const s = await scoreCase(c);
    console.log(`    aggregate=${s.aggregate.toFixed(4)} minOfMax=${s.minOfMax.toFixed(4)}`);
    negAgg.push(s.aggregate);
    negMinMax.push(s.minOfMax);
  }

  console.log('\n\n=================== RESULTS ===================');
  console.log('\nAggregate score (mean of best-match-per-load-image):');
  console.log('  positives:', stats(posAgg));
  console.log('  negatives:', stats(negAgg));
  asciiHistogram(posAgg, 'positives aggregate');
  asciiHistogram(negAgg, 'negatives aggregate');

  console.log('\nMin-of-max score (worst load image best match - the "smuggled pallet" signal):');
  console.log('  positives:', stats(posMinMax));
  console.log('  negatives:', stats(negMinMax));
  asciiHistogram(posMinMax, 'positives min-of-max');
  asciiHistogram(negMinMax, 'negatives min-of-max');

  // -------------------------------------------------------------------
  // Threshold recommendations.
  //
  //   PASS aggregate threshold:   pick a value that LET'S MOST POSITIVES THROUGH
  //                               -> P10 of positives is a good starting point
  //                               -> 90% of true positives clear it
  //   SUSPICIOUS aggregate:       pick a value where MOST NEGATIVES FALL BELOW
  //                               -> P90 of negatives means 90% of fraud caught
  //   minOfMax red flag:          P95 of negatives' minOfMax - very conservative,
  //                               only triggers on genuinely off-looking single
  //                               images
  // -------------------------------------------------------------------
  const recPass = percentile(posAgg, 10);
  const recSus = percentile(negAgg, 90);
  const recRedFlag = percentile(negMinMax, 95);

  console.log('\n=================== RECOMMENDED THRESHOLDS ===================');
  console.log('Copy these into your .env (or refine based on the histograms above):');
  console.log('');
  console.log(`  THRESHOLD_PASS_AGGREGATE=${recPass.toFixed(3)}`);
  console.log(`  THRESHOLD_SUSPICIOUS_AGGREGATE=${recSus.toFixed(3)}`);
  console.log(`  THRESHOLD_MIN_OF_MAX_RED_FLAG=${recRedFlag.toFixed(3)}`);
  console.log('');
  console.log('Sanity checks:');
  console.log(`  - pass threshold (${recPass.toFixed(3)}) should be > suspicious threshold (${recSus.toFixed(3)})`);
  if (recPass <= recSus) {
    console.log('  WARNING: pass <= suspicious. Your distributions overlap badly. Consider:');
    console.log('    1. More fixture data (50+50 minimum recommended)');
    console.log('    2. dinov2-large instead of dinov2-base (higher discrimination)');
    console.log('    3. Reviewing failure modes in negatives that scored high - they may not be true negatives');
  }
  console.log(`  - all-pass count: positives above pass = ${posAgg.filter(x => x >= recPass).length}/${posAgg.length}`);
  console.log(`  - all-fail count: negatives below suspicious = ${negAgg.filter(x => x <= recSus).length}/${negAgg.length}`);

  process.exit(0);
}

main().catch(err => {
  console.error('calibration failed:', err);
  process.exit(1);
});
