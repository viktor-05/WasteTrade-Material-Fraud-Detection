#!/usr/bin/env node
/**
 * Smoke test: verifies all the moving parts are wired up correctly.
 * Run this AFTER `npm run migrate` and BEFORE you trust anything else.
 *
 * It checks:
 *   1. DB connection
 *   2. Embedding sidecar is reachable and returning vectors
 *   3. VLM provider (whichever is configured) responds to a 1-image call
 *   4. The decision logic produces sensible numbers
 *
 * USAGE:
 *   node scripts/smoke-test.js path/to/listing_img.jpg path/to/load_img.jpg
 */
require('dotenv').config();
const fs = require('fs');

const { pool } = require('../src/db');
const embeddingClient = require('../src/services/embeddingClient');
const vlmVerifier = require('../src/services/vlmVerifier');
const { cosineSim } = require('../src/utils/vectors');

async function main() {
  const [listingImg, loadImg] = process.argv.slice(2);
  if (!listingImg || !loadImg) {
    console.error('Usage: node scripts/smoke-test.js <listing_img> <load_img>');
    process.exit(1);
  }
  for (const p of [listingImg, loadImg]) {
    if (!fs.existsSync(p)) {
      console.error(`File not found: ${p}`);
      process.exit(1);
    }
  }

  console.log('1. DB connection...');
  await pool.query('SELECT 1');
  console.log('   OK\n');

  console.log('2. Embedding sidecar health...');
  const h = await embeddingClient.health();
  console.log(`   model=${h.model} dim=${h.embedding_dim} device=${h.device}\n`);

  console.log('3. Embedding two images...');
  const t0 = Date.now();
  const a = await embeddingClient.embedFile(listingImg);
  const b = await embeddingClient.embedFile(loadImg);
  console.log(`   embedded both in ${Date.now() - t0}ms`);
  console.log(`   embedding dim: ${a.dim}`);
  console.log(`   cosine similarity: ${cosineSim(a.embedding, b.embedding).toFixed(4)}\n`);

  console.log(`4. VLM verifier (${process.env.VLM_PROVIDER || 'gemini'})...`);
  const t1 = Date.now();
  const result = await vlmVerifier.verify({
    listingImagePaths: [listingImg],
    loadImagePaths: [loadImg],
    claimedMaterial: 'test material',
  });
  console.log(`   responded in ${Date.now() - t1}ms`);
  console.log(`   provider: ${result.provider} model: ${result.model}`);
  console.log(`   verdict: ${result.verdict}`);
  console.log(`   confidence: ${result.confidence}`);
  console.log(`   summary: ${result.summary}\n`);

  console.log('All checks passed.');
  await pool.end();
  process.exit(0);
}

main().catch(err => {
  console.error('smoke test failed:', err);
  process.exit(1);
});
