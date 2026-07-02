import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Resolve Gateway auth token for local bench runs.
 * Priority: explicit option/env > openclaw.json gateway.auth.token
 */
export async function resolveGatewayToken(explicitToken) {
  const fromEnv = process.env.OPENCLAW_GATEWAY_TOKEN?.trim();
  if (explicitToken?.trim()) {
    return { token: explicitToken.trim(), source: "cli" };
  }
  if (fromEnv) {
    return { token: fromEnv, source: "env:OPENCLAW_GATEWAY_TOKEN" };
  }

  const configPath =
    process.env.OPENCLAW_CONFIG_PATH?.trim() ||
    join(process.env.OPENCLAW_STATE_DIR?.trim() || join(homedir(), ".openclaw"), "openclaw.json");

  try {
    const raw = await readFile(configPath, "utf8");
    const config = JSON.parse(raw);
    const token = config?.gateway?.auth?.token;
    if (typeof token === "string" && token.trim()) {
      return { token: token.trim(), source: `config:${configPath}` };
    }
  } catch {
    // fall through
  }

  return { token: "", source: "none" };
}

export function formatTokenMismatchHelp() {
  return (
    "Gateway token mismatch. The token sent by the bench driver does not match " +
    "gateway.auth.token in the openclaw.json used by the running Gateway process.\n" +
    "Fix:\n" +
    "  1. Unset a wrong OPENCLAW_GATEWAY_TOKEN (e.g. export OPENCLAW_GATEWAY_TOKEN=)\n" +
    "  2. Or set the exact token from the same config the Gateway loaded:\n" +
    "       grep -A2 '\"auth\"' ~/.openclaw/openclaw.json\n" +
    "  3. Or omit OPENCLAW_GATEWAY_TOKEN — the driver auto-reads ~/.openclaw/openclaw.json"
  );
}

export async function readOpenClawConfig() {
  const configPath =
    process.env.OPENCLAW_CONFIG_PATH?.trim() ||
    join(process.env.OPENCLAW_STATE_DIR?.trim() || join(homedir(), ".openclaw"), "openclaw.json");
  try {
    const raw = await readFile(configPath, "utf8");
    return { configPath, config: JSON.parse(raw) };
  } catch {
    return { configPath, config: null };
  }
}

const BENCH_GATEWAY_TOOLS = ["sessions_spawn"];

export function gatewayAllowsBenchTools(config) {
  const allow = config?.gateway?.tools?.allow;
  if (!Array.isArray(allow)) {
    return { ok: false, missing: BENCH_GATEWAY_TOOLS };
  }
  const allowed = new Set(allow.map((name) => String(name).trim()));
  const missing = BENCH_GATEWAY_TOOLS.filter((name) => !allowed.has(name));
  return { ok: missing.length === 0, missing };
}

export function formatSessionsSpawnBlockedHelp(configPath) {
  return (
    "sessions_spawn is blocked for Gateway tools.invoke (HTTP surface deny list).\n" +
    "OpenClaw denies sessions_spawn over tools.invoke by default for security.\n" +
    "Add to the openclaw.json used by your running Gateway, then restart gateway:\n\n" +
    '  "gateway": {\n' +
    '    "tools": {\n' +
    '      "allow": ["sessions_spawn"]\n' +
    "    }\n" +
    "  }\n\n" +
    `Config file: ${configPath}\n` +
    "Only enable this on loopback/local bench hosts."
  );
}

export async function assertBenchGatewayConfig() {
  const { configPath, config } = await readOpenClawConfig();
  const gate = gatewayAllowsBenchTools(config);
  if (!gate.ok) {
    throw new Error(formatSessionsSpawnBlockedHelp(configPath));
  }
  return { configPath, config };
}

export function assertKvcommSidecarGatewayModel(config, { configPath, model } = {}) {
  const inferenceBackend = (process.env.KVCOMM_INFERENCE_BACKEND || "").trim();
  if (inferenceBackend && inferenceBackend !== "kvcomm_sidecar") {
    return;
  }
  const primary = config?.agents?.defaults?.model?.primary;
  const providers = config?.models?.providers || {};
  const kvcommUrl = providers.kvcomm?.baseUrl || "";
  const wantsKvcommModel = !model || String(model).startsWith("kvcomm/");
  if (!wantsKvcommModel) {
    return;
  }
  if (primary && String(primary).startsWith("vllm/")) {
    throw new Error(
      `OpenClaw primary model is ${primary} but bench uses kvcomm_sidecar.\n` +
        `Run: node cli.mjs setup clawbench-capability-sidecar\n` +
        `Then restart gateway and use --model kvcomm/Qwen3-32B\n` +
        `Config: ${configPath}`,
    );
  }
  if (!kvcommUrl.includes("8100") && !process.env.KVCOMM_SIDECAR_URL?.trim()) {
    console.warn(
      `[clawbench-chain] warning: kvcomm provider baseUrl=${kvcommUrl || "(missing)"} ` +
        `may not point at sidecar (expected http://127.0.0.1:8100/v1)`,
    );
  }
}

const DEFAULT_CAPABILITY_SUBAGENT_TOOLS = [
  "read",
  "write",
  "edit",
  "apply_patch",
  "exec",
  "process",
  "browser",
  "session_status",
];

const CAPABILITY_ACTION_TOOLS = new Set([
  "read",
  "write",
  "edit",
  "apply_patch",
  "exec",
  "process",
  "browser",
]);

/** tools.subagents.tools.allow from openclaw.json (ClawBench capability lane). */
export async function resolveCapabilitySubagentTools() {
  const { configPath, config } = await readOpenClawConfig();
  const allow = config?.tools?.subagents?.tools?.allow;
  if (!Array.isArray(allow) || allow.length === 0) {
    return DEFAULT_CAPABILITY_SUBAGENT_TOOLS;
  }
  const normalized = allow.map((name) => String(name).trim()).filter(Boolean);
  const hasActionTools = normalized.some((name) => CAPABILITY_ACTION_TOOLS.has(name));
  if (!hasActionTools) {
    console.warn(
      `[clawbench-chain] tools.subagents.tools.allow=${JSON.stringify(normalized)} ` +
        `in ${configPath} lacks action tools (read/edit/exec). ` +
        `Using default capability set for sessions.patch inheritedToolAllow. ` +
        `Run: node cli.mjs setup clawbench-capability-sidecar && restart gateway.`,
    );
    return DEFAULT_CAPABILITY_SUBAGENT_TOOLS;
  }
  const merged = [...normalized];
  for (const tool of DEFAULT_CAPABILITY_SUBAGENT_TOOLS) {
    if (!merged.includes(tool)) {
      merged.push(tool);
    }
  }
  if (!normalized.includes("browser") && merged.includes("browser")) {
    console.warn(
      `[clawbench-chain] tools.subagents.tools.allow in ${configPath} is missing browser. ` +
        `Patching inheritedToolAllow with browser anyway. ` +
        `Run: node cli.mjs setup clawbench-capability-sidecar && restart gateway.`,
    );
  }
  return merged;
}

export async function assertCapabilitySubagentTools() {
  const { configPath, config } = await readOpenClawConfig();
  const allow = config?.tools?.subagents?.tools?.allow;
  const normalized = Array.isArray(allow)
    ? allow.map((name) => String(name).trim()).filter(Boolean)
    : [];
  const hasActionTools = normalized.some((name) => CAPABILITY_ACTION_TOOLS.has(name));
  if (hasActionTools) {
    return { configPath, allow: normalized };
  }
  throw new Error(
    "ClawBench capability lane requires subagent action tools (read/edit/exec).\n" +
      `Current tools.subagents.tools.allow=${JSON.stringify(normalized)} in ${configPath}\n` +
      "Fix:\n" +
      "  cd /src/KVCOMM/openclaw && ./scripts/setup-openclaw.sh clawbench-capability-sidecar\n" +
      "  openclaw gateway stop && openclaw gateway run\n" +
      "Bench will still patch inheritedToolAllow with the default set, but gateway policy should match.",
  );
}
