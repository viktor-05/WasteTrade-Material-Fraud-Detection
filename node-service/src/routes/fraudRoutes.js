/**
 * HTTP routes for the fraud-detection feature.
 *
 * Endpoints:
 *   POST   /listings                          - create listing + upload images
 *   POST   /listings/:id/images               - add more images to a listing
 *   GET    /listings/:id                      - listing + image metadata
 *
 *   POST   /loads                             - create load + upload images
 *   GET    /loads/:id                         - load + image metadata
 *
 *   POST   /fraud-check                       - run fraud check (listing_id, load_id)
 *   GET    /fraud-check/:id                   - retrieve a stored check
 *   POST   /fraud-check/:id/replay            - re-decide vs current thresholds
 *   POST   /fraud-check/:id/outcome           - reviewer label (for calibration)
 *   GET    /fraud-check                       - list checks (paginated)
 */
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const { pool } = require('../db');
const { runFraudCheck, replayFraudCheck } = require('../services/fraudCheck');

const router = express.Router();

// ---------------------------------------------------------------------------
// File upload setup. Stores to IMAGE_STORAGE_DIR with deterministic-ish names
// so paths in MySQL stay stable.
// ---------------------------------------------------------------------------
const STORAGE_DIR = process.env.IMAGE_STORAGE_DIR || path.join(__dirname, '../../uploads');
fs.mkdirSync(STORAGE_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const sub = path.join(STORAGE_DIR, req.params.kind || req.body.kind || 'misc');
    fs.mkdirSync(sub, { recursive: true });
    cb(null, sub);
  },
  filename: (req, file, cb) => {
    const random = crypto.randomBytes(8).toString('hex');
    const ext = path.extname(file.originalname).toLowerCase() || '.jpg';
    cb(null, `${Date.now()}-${random}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB per file - matches sidecar limit
  fileFilter: (_req, file, cb) => {
    const ok = /^image\/(jpeg|png|webp|gif)$/i.test(file.mimetype);
    cb(ok ? null : new Error(`Unsupported file type: ${file.mimetype}`), ok);
  },
});

// ---------------------------------------------------------------------------
// Listings
// ---------------------------------------------------------------------------
router.post('/listings', upload.array('images', 10), async (req, res, next) => {
  try {
    const { material_type } = req.body;
    if (!material_type) return res.status(400).json({ error: 'material_type is required' });
    if (!req.files?.length) return res.status(400).json({ error: 'At least one image is required' });

    const [result] = await pool.query(
      'INSERT INTO listings (material_type) VALUES (?)',
      [material_type]
    );
    const listingId = result.insertId;

    // Insert image rows. Embeddings filled lazily on first fraud check.
    const values = req.files.map(f => [listingId, f.path, f.filename]);
    await pool.query(
      'INSERT INTO listing_images (listing_id, path, filename) VALUES ?',
      [values]
    );

    res.status(201).json({
      listing_id: listingId,
      material_type,
      images_uploaded: req.files.length,
    });
  } catch (err) { next(err); }
});

router.post('/listings/:id/images', upload.array('images', 10), async (req, res, next) => {
  try {
    const listingId = parseInt(req.params.id, 10);
    if (!req.files?.length) return res.status(400).json({ error: 'At least one image is required' });

    const values = req.files.map(f => [listingId, f.path, f.filename]);
    await pool.query(
      'INSERT INTO listing_images (listing_id, path, filename) VALUES ?',
      [values]
    );
    res.json({ listing_id: listingId, images_uploaded: req.files.length });
  } catch (err) { next(err); }
});

router.get('/listings/:id', async (req, res, next) => {
  try {
    const listingId = parseInt(req.params.id, 10);
    const [listings] = await pool.query('SELECT * FROM listings WHERE id = ?', [listingId]);
    if (!listings.length) return res.status(404).json({ error: 'Listing not found' });
    const [images] = await pool.query(
      'SELECT id, filename, path, embedding_model, created_at FROM listing_images WHERE listing_id = ?',
      [listingId]
    );
    res.json({ ...listings[0], images });
  } catch (err) { next(err); }
});

// ---------------------------------------------------------------------------
// Loads
// ---------------------------------------------------------------------------
router.post('/loads', upload.array('images', 10), async (req, res, next) => {
  try {
    const { listing_id } = req.body;
    if (!listing_id) return res.status(400).json({ error: 'listing_id is required' });
    if (!req.files?.length) return res.status(400).json({ error: 'At least one image is required' });

    const [result] = await pool.query(
      'INSERT INTO loads (listing_id) VALUES (?)',
      [listing_id]
    );
    const loadId = result.insertId;

    const values = req.files.map(f => [loadId, f.path, f.filename]);
    await pool.query(
      'INSERT INTO load_images (load_id, path, filename) VALUES ?',
      [values]
    );

    res.status(201).json({ load_id: loadId, listing_id: parseInt(listing_id, 10), images_uploaded: req.files.length });
  } catch (err) { next(err); }
});

router.get('/loads/:id', async (req, res, next) => {
  try {
    const loadId = parseInt(req.params.id, 10);
    const [loads] = await pool.query('SELECT * FROM loads WHERE id = ?', [loadId]);
    if (!loads.length) return res.status(404).json({ error: 'Load not found' });
    const [images] = await pool.query(
      'SELECT id, filename, path, embedding_model, created_at FROM load_images WHERE load_id = ?',
      [loadId]
    );
    res.json({ ...loads[0], images });
  } catch (err) { next(err); }
});

// ---------------------------------------------------------------------------
// Fraud checks
// ---------------------------------------------------------------------------
router.post('/fraud-check', async (req, res, next) => {
  try {
    const { listing_id, load_id, claimed_material, vlm_override } = req.body;
    if (!listing_id || !load_id) {
      return res.status(400).json({ error: 'listing_id and load_id are required' });
    }

    let claimed = claimed_material;
    if (!claimed) {
      const [r] = await pool.query('SELECT material_type FROM listings WHERE id = ?', [listing_id]);
      claimed = r[0]?.material_type || null;
    }

    const result = await runFraudCheck({
      listingId: parseInt(listing_id, 10),
      loadId: parseInt(load_id, 10),
      claimedMaterial: claimed,
      vlmOverride: vlm_override,
    });
    res.json(result);
  } catch (err) { next(err); }
});

router.get('/fraud-check/:id', async (req, res, next) => {
  try {
    const id = parseInt(req.params.id, 10);
    const [rows] = await pool.query('SELECT * FROM fraud_checks WHERE id = ?', [id]);
    if (!rows.length) return res.status(404).json({ error: 'Fraud check not found' });
    res.json(rows[0]);
  } catch (err) { next(err); }
});

router.post('/fraud-check/:id/replay', async (req, res, next) => {
  try {
    const id = parseInt(req.params.id, 10);
    const result = await replayFraudCheck(id);
    res.json(result);
  } catch (err) { next(err); }
});

/**
 * Reviewer outcome submission. This is the calibration goldmine - every
 * labeled outcome here makes future threshold tuning more accurate.
 */
router.post('/fraud-check/:id/outcome', async (req, res, next) => {
  try {
    const id = parseInt(req.params.id, 10);
    const { outcome, reviewer_notes } = req.body;
    const allowed = ['confirmed_fraud', 'false_alarm', 'inconclusive'];
    if (!allowed.includes(outcome)) {
      return res.status(400).json({ error: `outcome must be one of ${allowed.join(', ')}` });
    }

    await pool.query(
      'UPDATE fraud_checks SET reviewer_outcome = ?, reviewer_notes = ?, reviewed_at = NOW() WHERE id = ?',
      [outcome, reviewer_notes || null, id]
    );
    res.json({ ok: true });
  } catch (err) { next(err); }
});

router.get('/fraud-check', async (req, res, next) => {
  try {
    const limit = Math.min(parseInt(req.query.limit || '50', 10), 200);
    const offset = parseInt(req.query.offset || '0', 10);
    const decision = req.query.decision; // optional filter

    let sql = `SELECT id, listing_id, load_id, claimed_material, aggregate_score,
                      min_of_max, vlm_verdict, vlm_confidence, decision,
                      reviewer_outcome, created_at, reviewed_at
               FROM fraud_checks`;
    const params = [];
    if (decision) {
      sql += ' WHERE decision = ?';
      params.push(decision);
    }
    sql += ' ORDER BY id DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    const [rows] = await pool.query(sql, params);
    res.json({ items: rows, limit, offset });
  } catch (err) { next(err); }
});

module.exports = router;
