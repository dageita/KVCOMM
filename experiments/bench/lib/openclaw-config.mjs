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
