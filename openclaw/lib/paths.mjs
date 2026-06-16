import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** KVCOMM repo root (…/KVCOMM). */
export const KVCOMM_ROOT = resolve(__dirname, "../..");

/** Canonical OpenClaw integration module root (…/KVCOMM/openclaw). */
export const OPENCLAW_MODULE_ROOT = resolve(KVCOMM_ROOT, "openclaw");

/** Legacy bench spike; implementation lives here until fully merged. */
export const BENCH_ROOT = resolve(KVCOMM_ROOT, "experiments/bench");

export function benchPath(...segments) {
  return join(BENCH_ROOT, ...segments);
}

export function configPath(...segments) {
  return join(OPENCLAW_MODULE_ROOT, "config", ...segments);
}

export function scriptPath(...segments) {
  return join(OPENCLAW_MODULE_ROOT, "scripts", ...segments);
}
