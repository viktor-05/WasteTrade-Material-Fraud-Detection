/**
 * Client for the DINOv2 embedding sidecar running on Railway.
 *
 * Two flows:
 *   - embedFile(path)  -> uploads the local file as multipart/form-data
 *   - embedUrl(url)    -> sends a JSON request asking the sidecar to fetch it
 *
 * For WasteTrade where images live on the local server, embedFile is the
 * primary path. embedUrl exists for the case where images get moved to a
 * CDN later (you won't need to change anything here).
 */
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');

const BASE_URL = (process.env.EMBEDDING_SERVICE_URL || '').replace(/\/$/, '');
const API_KEY = process.env.EMBEDDING_API_KEY || '';
const MODEL_VERSION = process.env.EMBEDDING_MODEL_VERSION || 'dinov2-base-v1';
const TIMEOUT_MS = 60_000; // first request after sleep can take a while

function _assertConfigured() {
  if (!BASE_URL) {
    throw new Error('EMBEDDING_SERVICE_URL is not set. Configure your Railway URL in .env.');
  }
}

/**
 * Embed a single local file.
 * Returns { embedding: number[], dim, model, modelVersion }.
 */
async function embedFile(filePath) {
  _assertConfigured();

  if (!fs.existsSync(filePath)) {
    throw new Error(`Image file not found: ${filePath}`);
  }

  const form = new FormData();
  form.append('image', fs.createReadStream(filePath), path.basename(filePath));
  if (API_KEY) form.append('x_api_key', API_KEY);

  const resp = await axios.post(`${BASE_URL}/embed`, form, {
    headers: form.getHeaders(),
    timeout: TIMEOUT_MS,
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
  });

  return {
    embedding: resp.data.embedding,
    dim: resp.data.dim,
    model: resp.data.model,
    modelVersion: MODEL_VERSION,
  };
}

/**
 * Embed an image by URL (sidecar fetches it).
 */
async function embedUrl(imageUrl) {
  _assertConfigured();

  const resp = await axios.post(
    `${BASE_URL}/embed/json`,
    { image_url: imageUrl },
    {
      params: API_KEY ? { x_api_key: API_KEY } : undefined,
      timeout: TIMEOUT_MS,
    }
  );

  return {
    embedding: resp.data.embedding,
    dim: resp.data.dim,
    model: resp.data.model,
    modelVersion: MODEL_VERSION,
  };
}

/**
 * Health check - useful for the Node app's own /health endpoint to
 * report whether the sidecar is reachable.
 */
async function health() {
  _assertConfigured();
  const resp = await axios.get(`${BASE_URL}/health`, { timeout: 10_000 });
  return resp.data;
}

module.exports = {
  embedFile,
  embedUrl,
  health,
  MODEL_VERSION,
};
