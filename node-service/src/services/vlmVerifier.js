/**
 * VLM verifier - the second signal next to embeddings.
 *
 * Two providers are wired up; switch with VLM_PROVIDER=gemini|claude.
 *
 * The prompt design is the most important thing in this file. Two rules
 * we enforce that VLMs love to violate:
 *
 *   1. BLIND DESCRIPTION FIRST. The model describes set A and set B
 *      independently, BEFORE seeing the comparison instruction. This stops
 *      the "the listing says LDPE so I'll find LDPE in the load photos"
 *      anchoring effect. Without this step both Gemini and Claude tend to
 *      rationalize away real discrepancies.
 *
 *   2. STRICT JSON OUTPUT. {verdict, confidence, discrepancies[], summary}.
 *      Both providers support this differently:
 *        - Gemini: response_schema + response_mime_type='application/json'
 *        - Claude: tool-use forces a structured payload
 */
const fs = require('fs');
const path = require('path');

const PROVIDER = (process.env.VLM_PROVIDER || 'gemini').toLowerCase();
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-4-5';

const VERDICT_VALUES = ['match', 'mismatch', 'uncertain'];

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------
const SYSTEM_PROMPT = `You are an expert waste-materials inspector for a global recyclables marketplace.
Your job is to verify whether the LOAD photos from a shipment match the LISTING photos
they were sold under.

You will be shown two sets of images:
  - SET A: the listing photos the buyer agreed to buy
  - SET B: the load photos showing what was actually shipped

Follow these steps in order. Do NOT skip step 1 or step 2.

  Step 1: Describe SET A independently. List the polymer/material type if
          identifiable, dominant colors, bale format and packaging, surface
          texture, contamination level, and apparent scale.
  Step 2: Describe SET B independently. Use the SAME categories. CRITICAL:
          do not assume SET B contains the same material as SET A. Describe
          only what you can see.
  Step 3: Compare them on:
            - Polymer / material family
            - Color / grade
            - Bale form, size, and binding
            - Contamination type and level
            - Scale and packaging context
  Step 4: Issue a verdict.

Account for benign differences: lighting (warehouse vs container vs daylight),
camera angle, distance, and image compression. These should NOT trigger a
mismatch by themselves.

Output STRICT JSON matching this schema, with no markdown fencing:
  {
    "set_a_description": string,
    "set_b_description": string,
    "comparison": string,
    "verdict": "match" | "mismatch" | "uncertain",
    "confidence": number (0.0 to 1.0),
    "discrepancies": [string, ...],
    "summary": string
  }

Verdict guidance:
  - "match"      : same material family AND same broad grade. High confidence.
  - "mismatch"   : different material, wrong color/grade, or visible contamination
                   that changes the product class.
  - "uncertain"  : insufficient image quality, conflicting signals, or you cannot
                   discriminate the relevant grades from photos alone.`;

function _buildUserPrompt(claimedMaterial) {
  const materialLine = claimedMaterial
    ? `The seller listed this material as: "${claimedMaterial}". Use this only as context AFTER you have described both sets blindly.`
    : `No claimed material was provided.`;

  return `${materialLine}

I will now show you SET A (listing photos) followed by SET B (load photos).
Please follow the four steps and return the JSON object.`;
}

// ---------------------------------------------------------------------------
// Image loading helpers
// ---------------------------------------------------------------------------
function _loadImageBase64(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Image file not found: ${filePath}`);
  }
  const ext = path.extname(filePath).toLowerCase().replace('.', '');
  const mimeMap = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp', gif: 'image/gif' };
  const mimeType = mimeMap[ext] || 'image/jpeg';
  const data = fs.readFileSync(filePath).toString('base64');
  return { mimeType, data };
}

// ---------------------------------------------------------------------------
// Gemini implementation
// ---------------------------------------------------------------------------
async function _verifyWithGemini({ listingImagePaths, loadImagePaths, claimedMaterial }) {
  if (!GEMINI_API_KEY) {
    throw new Error('GEMINI_API_KEY is not set.');
  }

  const { GoogleGenerativeAI, SchemaType } = require('@google/generative-ai');
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

  const responseSchema = {
    type: SchemaType.OBJECT,
    properties: {
      set_a_description: { type: SchemaType.STRING },
      set_b_description: { type: SchemaType.STRING },
      comparison: { type: SchemaType.STRING },
      verdict: { type: SchemaType.STRING, enum: VERDICT_VALUES },
      confidence: { type: SchemaType.NUMBER },
      discrepancies: { type: SchemaType.ARRAY, items: { type: SchemaType.STRING } },
      summary: { type: SchemaType.STRING },
    },
    required: ['set_a_description', 'set_b_description', 'comparison', 'verdict', 'confidence', 'discrepancies', 'summary'],
  };

  const model = genAI.getGenerativeModel({
    model: GEMINI_MODEL,
    systemInstruction: SYSTEM_PROMPT,
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema,
      temperature: 0.1, // we want consistent verdicts, not creative ones
    },
  });

  // Build a single user turn: prompt text, then label-image-label-image...
  const parts = [{ text: _buildUserPrompt(claimedMaterial) }];

  parts.push({ text: '\n--- SET A: LISTING PHOTOS ---\n' });
  listingImagePaths.forEach((p, i) => {
    const { mimeType, data } = _loadImageBase64(p);
    parts.push({ text: `Listing photo ${i + 1}:` });
    parts.push({ inlineData: { mimeType, data } });
  });

  parts.push({ text: '\n--- SET B: LOAD PHOTOS ---\n' });
  loadImagePaths.forEach((p, i) => {
    const { mimeType, data } = _loadImageBase64(p);
    parts.push({ text: `Load photo ${i + 1}:` });
    parts.push({ inlineData: { mimeType, data } });
  });

  const result = await model.generateContent({
    contents: [{ role: 'user', parts }],
  });

  const text = result.response.text();
  const parsed = JSON.parse(text); // schema-enforced, this should never throw

  return {
    provider: 'gemini',
    model: GEMINI_MODEL,
    ...parsed,
    rawResponse: text,
  };
}

// ---------------------------------------------------------------------------
// Claude implementation
// ---------------------------------------------------------------------------
async function _verifyWithClaude({ listingImagePaths, loadImagePaths, claimedMaterial }) {
  if (!ANTHROPIC_API_KEY) {
    throw new Error('ANTHROPIC_API_KEY is not set.');
  }

  const Anthropic = require('@anthropic-ai/sdk');
  const client = new Anthropic.default({ apiKey: ANTHROPIC_API_KEY });

  // Claude doesn't have native JSON schema enforcement on outputs. The
  // standard pattern is to expose a single tool and force the model to use
  // it - the tool's input_schema then validates the output structure.
  const verdictTool = {
    name: 'submit_verification',
    description: 'Submit the structured verification result for the load vs listing comparison.',
    input_schema: {
      type: 'object',
      properties: {
        set_a_description: { type: 'string' },
        set_b_description: { type: 'string' },
        comparison: { type: 'string' },
        verdict: { type: 'string', enum: VERDICT_VALUES },
        confidence: { type: 'number', minimum: 0, maximum: 1 },
        discrepancies: { type: 'array', items: { type: 'string' } },
        summary: { type: 'string' },
      },
      required: ['set_a_description', 'set_b_description', 'comparison', 'verdict', 'confidence', 'discrepancies', 'summary'],
    },
  };

  const content = [{ type: 'text', text: _buildUserPrompt(claimedMaterial) }];

  content.push({ type: 'text', text: '\n--- SET A: LISTING PHOTOS ---\n' });
  listingImagePaths.forEach((p, i) => {
    const { mimeType, data } = _loadImageBase64(p);
    content.push({ type: 'text', text: `Listing photo ${i + 1}:` });
    content.push({ type: 'image', source: { type: 'base64', media_type: mimeType, data } });
  });

  content.push({ type: 'text', text: '\n--- SET B: LOAD PHOTOS ---\n' });
  loadImagePaths.forEach((p, i) => {
    const { mimeType, data } = _loadImageBase64(p);
    content.push({ type: 'text', text: `Load photo ${i + 1}:` });
    content.push({ type: 'image', source: { type: 'base64', media_type: mimeType, data } });
  });

  const response = await client.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 2048,
    system: SYSTEM_PROMPT,
    tools: [verdictTool],
    tool_choice: { type: 'tool', name: 'submit_verification' },
    messages: [{ role: 'user', content }],
    temperature: 0.1,
  });

  const toolUse = response.content.find(b => b.type === 'tool_use');
  if (!toolUse) {
    throw new Error('Claude did not return a tool_use block.');
  }

  return {
    provider: 'claude',
    model: CLAUDE_MODEL,
    ...toolUse.input,
    rawResponse: JSON.stringify(toolUse.input),
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
/**
 * Run the VLM verification.
 * @param {object} params
 * @param {string[]} params.listingImagePaths - absolute paths to listing photos
 * @param {string[]} params.loadImagePaths    - absolute paths to load photos
 * @param {string}   [params.claimedMaterial] - material label from the listing
 * @param {string}   [params.providerOverride] - 'gemini' | 'claude' to override env
 */
async function verify({ listingImagePaths, loadImagePaths, claimedMaterial, providerOverride }) {
  if (!listingImagePaths?.length || !loadImagePaths?.length) {
    throw new Error('Both listingImagePaths and loadImagePaths must be non-empty.');
  }

  const provider = (providerOverride || PROVIDER).toLowerCase();

  switch (provider) {
    case 'gemini':
      return _verifyWithGemini({ listingImagePaths, loadImagePaths, claimedMaterial });
    case 'claude':
      return _verifyWithClaude({ listingImagePaths, loadImagePaths, claimedMaterial });
    default:
      throw new Error(`Unknown VLM_PROVIDER: ${provider}. Use 'gemini' or 'claude'.`);
  }
}

module.exports = {
  verify,
  PROVIDER,
  VERDICT_VALUES,
};
