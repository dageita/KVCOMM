/**
 * Replace {{key}} placeholders in task templates.
 */
export function renderTemplate(template, variables) {
  if (typeof template !== "string") {
    throw new TypeError("template must be a string");
  }
  return template.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (match, key) => {
    if (!Object.hasOwn(variables, key)) {
      return match;
    }
    const value = variables[key];
    return value == null ? "" : String(value);
  });
}

export function renderTemplateStrict(template, variables) {
  if (typeof template !== "string") {
    throw new TypeError("template must be a string");
  }
  return template.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_, key) => {
    if (!Object.hasOwn(variables, key)) {
      throw new Error(`Missing template variable: ${key}`);
    }
    const value = variables[key];
    return value == null ? "" : String(value);
  });
}

/** KVCOMM placeholder keys kept as Python-style `{agent_N_current}` for sidecar kv_reuse. */
const KVCOMM_PLACEHOLDER_KEYS = new Set(
  Array.from({ length: 16 }, (_, i) => `agent_${i}_current`).concat(
    Array.from({ length: 16 }, (_, i) => `agent_${i}_history`),
    ["user_question"],
  ),
);

/**
 * Render bench templates for kv_reuse: expand {{var}} except upstream agent placeholders,
 * which become `{agent_N_current}` for KVCOMM locate_placeholder.
 */
export function renderTemplateKvReuse(template, variables) {
  if (typeof template !== "string") {
    throw new TypeError("template must be a string");
  }
  return template.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (match, key) => {
    if (KVCOMM_PLACEHOLDER_KEYS.has(key)) {
      return `{${key}}`;
    }
    if (!Object.hasOwn(variables, key)) {
      throw new Error(`Missing template variable: ${key}`);
    }
    const value = variables[key];
    return value == null ? "" : String(value);
  });
}

export function buildKvcommMetaPrefix(meta) {
  return `<!--KVCOMM_META:${JSON.stringify(meta)}-->\n`;
}

export function sha256Short(text) {
  // Lightweight fingerprint for L2 verification (no crypto dep required for spike).
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
