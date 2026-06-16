/**
 * Detect Qwen / OpenClaw thinking output in assistant text.
 */

const THINK_OPEN = "<" + "think" + ">";
const THINK_CLOSE = "<" + "/think" + ">";
const REDACTED_OPEN = "<think>";
const REDACTED_CLOSE = "</think>";

export function detectThinkingInText(text) {
  if (!text || typeof text !== "string") {
    return false;
  }
  const lower = text.toLowerCase();
  return (
    lower.includes(THINK_OPEN)
    || lower.includes(THINK_CLOSE)
    || lower.includes(REDACTED_OPEN)
    || lower.includes(REDACTED_CLOSE)
    || lower.includes("reasoning_content")
  );
}
