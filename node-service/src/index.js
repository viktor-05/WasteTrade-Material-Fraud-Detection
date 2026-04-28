/**
 * Standalone Express app for the fraud detection MVP.
 *
 * If you want to mount this INTO your existing WasteTrade Express app
 * instead of running it standalone, just do:
 *
 *   const fraudRoutes = require('./path/to/fraudRoutes');
 *   yourApp.use('/fraud', fraudRoutes);
 *
 * That's the entire integration.
 */
require('dotenv').config();

const express = require('express');
const path = require('path');

const fraudRoutes = require('./routes/fraudRoutes');
const embeddingClient = require('./services/embeddingClient');

const app = express();
app.use(express.json({ limit: '1mb' }));

// Static serving of uploaded images, for review screen previews. Lock this
// down (auth/IP allowlist) before going to production.
const STORAGE_DIR = process.env.IMAGE_STORAGE_DIR || path.join(__dirname, '../uploads');
app.use('/uploads', express.static(STORAGE_DIR));

// Health: verifies DB and embedding sidecar are both reachable.
app.get('/health', async (_req, res) => {
  const result = { status: 'ok', service: 'wastetrade-fraud-detection' };
  try {
    const { pool } = require('./db');
    await pool.query('SELECT 1');
    result.db = 'ok';
  } catch (err) {
    result.db = `error: ${err.message}`;
    result.status = 'degraded';
  }
  try {
    const sidecar = await embeddingClient.health();
    result.embedding_sidecar = sidecar;
  } catch (err) {
    result.embedding_sidecar = `error: ${err.message}`;
    result.status = 'degraded';
  }
  res.status(result.status === 'ok' ? 200 : 503).json(result);
});

app.use('/api/fraud', fraudRoutes);

// Centralized error handler. Multer errors come back here too.
// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  console.error('[error]', err);
  const status = err.status || (err.code === 'LIMIT_FILE_SIZE' ? 413 : 500);
  res.status(status).json({ error: err.message || 'Internal server error' });
});

const PORT = parseInt(process.env.PORT || '3001', 10);
app.listen(PORT, () => {
  console.log(`[wastetrade-fraud-detection] listening on :${PORT}`);
  console.log(`[wastetrade-fraud-detection] storage dir: ${STORAGE_DIR}`);
  console.log(`[wastetrade-fraud-detection] vlm provider: ${process.env.VLM_PROVIDER || 'gemini'}`);
});
