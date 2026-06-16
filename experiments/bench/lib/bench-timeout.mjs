/**
 * Resolve per-agent OpenClaw run / stream wait timeout (seconds).
 *
 * Priority:
 *   1. explicit CLI --run-timeout-seconds
 *   2. BENCH_RUN_TIMEOUT_SECONDS env
 *   3. debug mode → 60s
 *   4. default → 600s
 */

const DEBUG_TRUTHY = new Set(["1", "true", "yes", "on"]);

export function isBenchDebugMode(env = process.env) {
  for (const key of ["BENCH_DEBUG", "KVCOMM_BENCH_DEBUG"]) {
    const raw = env[key]?.trim().toLowerCase();
    if (raw && DEBUG_TRUTHY.has(raw)) {
      return true;
    }
  }
  const loguru = env.LOGURU_LEVEL?.trim().toUpperCase();
  return loguru === "DEBUG";
}

export function resolveRunTimeoutSeconds({ explicitSeconds, debugFlag } = {}, env = process.env) {
  if (explicitSeconds != null) {
    const value = Number(explicitSeconds);
    if (Number.isFinite(value) && value > 0) {
      return Math.round(value);
    }
  }

  const fromEnv = env.BENCH_RUN_TIMEOUT_SECONDS?.trim();
  if (fromEnv) {
    const value = Number(fromEnv);
    if (Number.isFinite(value) && value > 0) {
      return Math.round(value);
    }
  }

  const debug = debugFlag === true || (debugFlag !== false && isBenchDebugMode(env));
  return debug ? 60 : 600;
}
