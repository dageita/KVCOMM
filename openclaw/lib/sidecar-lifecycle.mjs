/**
 * Managed KVCOMM sidecar lifecycle for bench runs:
 * - start lightweight proxy only when needed
 * - release HF weights after bench completes
 * - stop child process if this driver started it
 */

import { spawn, spawnSync } from "node:child_process";
import { OPENCLAW_MODULE_ROOT, KVCOMM_ROOT } from "./paths.mjs";
import { join } from "node:path";

const DEFAULT_SIDECAR_URL = process.env.KVCOMM_SIDECAR_URL?.trim() || "http://127.0.0.1:8100";

export function resolveSidecarPython() {
  if (process.env.KVCOMM_PYTHON?.trim()) {
    return process.env.KVCOMM_PYTHON.trim();
  }
  const candidates = [
    "/opt/conda/envs/kvcomm/bin/python3",
    "/opt/conda/envs/crius/bin/python3",
  ];
  for (const python of candidates) {
    const probe = spawnSync(python, ["-c", "import httpx, transformers"], { stdio: "ignore" });
    if (probe.status === 0) {
      return python;
    }
  }
  return "python3";
}

export function shouldManageSidecar(inferenceBackend) {
  const flag = (process.env.KVCOMM_MANAGE_SIDECAR ?? "auto").trim().toLowerCase();
  if (flag === "0" || flag === "false" || flag === "no") {
    return false;
  }
  if (flag === "1" || flag === "true" || flag === "yes") {
    return true;
  }
  return inferenceBackend === "kvcomm_sidecar";
}

export function shouldReleaseEngine() {
  const flag = (process.env.KVCOMM_RELEASE_ENGINE ?? "1").trim().toLowerCase();
  return !["0", "false", "no"].includes(flag);
}

function sidecarBaseUrl(sidecarUrl = DEFAULT_SIDECAR_URL) {
  return sidecarUrl.replace(/\/$/, "");
}

export async function fetchSidecarHealth(sidecarUrl = DEFAULT_SIDECAR_URL) {
  try {
    const resp = await fetch(`${sidecarBaseUrl(sidecarUrl)}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) {
      return null;
    }
    return await resp.json();
  } catch {
    return null;
  }
}

export async function waitForSidecarHealth(
  sidecarUrl = DEFAULT_SIDECAR_URL,
  { timeoutMs = 120_000, pollMs = 500 } = {},
) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const health = await fetchSidecarHealth(sidecarUrl);
    if (health?.status === "ok") {
      return health;
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
  throw new Error(`sidecar health check timed out after ${timeoutMs}ms (${sidecarUrl})`);
}

function buildBenchHfConfig() {
  const hfDevice = process.env.KVCOMM_HF_DEVICE?.trim() || "";
  const denseViaHf = (process.env.KVCOMM_DENSE_VIA_HF ?? "").trim().toLowerCase();
  const denseViaHfEnabled = ["1", "true", "yes", "on"].includes(denseViaHf);
  return {
    hf_model: process.env.KVCOMM_HF_MODEL?.trim() || "",
    hf_model_path: process.env.KVCOMM_HF_MODEL_PATH?.trim() || "",
    hf_device: hfDevice,
    cuda_visible_devices:
      process.env.CUDA_VISIBLE_DEVICES?.trim() ||
      process.env.KVCOMM_CUDA_VISIBLE_DEVICES?.trim() ||
      hfDevice ||
      "",
    dense_via_hf: denseViaHfEnabled,
    KVCOMM_DENSE_VIA_HF: denseViaHfEnabled ? "1" : "0",
  };
}

export async function configureSidecarEngine(
  sidecarUrl = DEFAULT_SIDECAR_URL,
  config = buildBenchHfConfig(),
) {
  if (!config.hf_model && !config.hf_model_path) {
    throw new Error(
      "KVCOMM_HF_MODEL or KVCOMM_HF_MODEL_PATH is required for kvcomm_sidecar bench runs",
    );
  }
  const url = `${sidecarBaseUrl(sidecarUrl)}/v1/kvcomm/configure`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
    signal: AbortSignal.timeout(30_000),
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`sidecar configure failed: ${resp.status} ${text.slice(0, 300)}`);
  }
  return await resp.json();
}

export async function releaseSidecarEngine(sidecarUrl = DEFAULT_SIDECAR_URL) {
  const url = `${sidecarBaseUrl(sidecarUrl)}/v1/kvcomm/release`;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(120_000),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      console.warn(`[sidecar] release failed: ${resp.status} ${text.slice(0, 200)}`);
      return null;
    }
    return await resp.json();
  } catch (err) {
    console.warn(`[sidecar] release error: ${err?.message ?? err}`);
    return null;
  }
}

function buildManagedSidecarEnv(sidecarUrl = DEFAULT_SIDECAR_URL) {
  let host = "127.0.0.1";
  let port = "8100";
  try {
    const parsed = new URL(sidecarUrl);
    host = parsed.hostname || host;
    port = parsed.port || port;
  } catch {
    // keep defaults
  }

  const hfDevice =
    process.env.KVCOMM_CUDA_VISIBLE_DEVICES?.trim() ||
    process.env.KVCOMM_HF_DEVICE?.trim() ||
    "";
  const env = {
    ...process.env,
    KVCOMM_SIDECAR_HOST: host,
    KVCOMM_SIDECAR_PORT: port,
    PYTHONPATH: [KVCOMM_ROOT, OPENCLAW_MODULE_ROOT, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(":"),
  };

  if (!env.KVCOMM_MODE?.trim()) {
    env.KVCOMM_MODE = "kv_reuse";
  }
  if (!env.KVCOMM_BENCH_NO_THINK?.trim()) {
    env.KVCOMM_BENCH_NO_THINK = "1";
  }
  const denseViaHf = (process.env.KVCOMM_DENSE_VIA_HF ?? "").trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(denseViaHf)) {
    env.KVCOMM_DENSE_VIA_HF = "1";
  }
  if (!env.KVCOMM_HF_MODEL?.trim() && !env.KVCOMM_HF_MODEL_PATH?.trim()) {
    env.KVCOMM_HF_MODEL = "/models/Qwen3-32B";
  }
  if (hfDevice && !env.CUDA_VISIBLE_DEVICES?.trim()) {
    const parts = hfDevice.split(",").map((part) => part.trim()).filter(Boolean);
    env.CUDA_VISIBLE_DEVICES = parts.join(",");
    env.KVCOMM_HF_DEVICE = parts.map((_, index) => String(index)).join(",");
  }
  return env;
}

export async function ensureManagedSidecarForBench({ inferenceBackend } = {}) {
  if (!shouldManageSidecar(inferenceBackend)) {
    return null;
  }

  const sidecarUrl = DEFAULT_SIDECAR_URL;
  const health = await fetchSidecarHealth(sidecarUrl);
  if (health?.status === "ok") {
    const benchConfig = buildBenchHfConfig();
    const hasHfModel = Boolean(health.hf_model || health.hf_model_env);
    const denseViaHf = (process.env.KVCOMM_DENSE_VIA_HF ?? "").trim().toLowerCase();
    const wantsDenseViaHf = ["1", "true", "yes", "on"].includes(denseViaHf);
    if (!hasHfModel) {
      const configured = await configureSidecarEngine(sidecarUrl, benchConfig);
      console.log(
        `[sidecar] configured lightweight sidecar (${sidecarUrl}) ` +
          `model=${configured.hf_model ?? "n/a"} engine_loaded=false (GPU on first kv_reuse) ` +
          `dense_via_hf=${wantsDenseViaHf}`,
      );
    } else if (benchConfig.hf_device && benchConfig.hf_device !== health.hf_device) {
      await configureSidecarEngine(sidecarUrl, benchConfig);
      console.log(`[sidecar] updated HF device pool on ${sidecarUrl}: ${benchConfig.hf_device}`);
    } else if (wantsDenseViaHf) {
      await configureSidecarEngine(sidecarUrl, benchConfig);
      console.log(`[sidecar] applied dense_via_hf=${wantsDenseViaHf} on ${sidecarUrl}`);
    } else {
      console.log(
        `[sidecar] reusing sidecar (${sidecarUrl}) engine_loaded=${health.engine_loaded ?? false}`,
      );
    }
    return { managed: false, sidecarUrl, child: null };
  }

  const python = resolveSidecarPython();
  const server = join(OPENCLAW_MODULE_ROOT, "sidecar/server.py");
  const env = buildManagedSidecarEnv(sidecarUrl);
  console.log(
    `[sidecar] starting managed sidecar on ${sidecarUrl} ` +
      `(HF model loads on first kv_reuse request; GPUs released after bench)`,
  );

  const child = spawn(python, [server], {
    env,
    cwd: OPENCLAW_MODULE_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) {
      console.log(`[sidecar] ${text}`);
    }
  });
  child.stderr?.on("data", (chunk) => {
    const text = chunk.toString().trim();
    if (text) {
      console.error(`[sidecar] ${text}`);
    }
  });

  await waitForSidecarHealth(sidecarUrl);
  const benchConfig = buildBenchHfConfig();
  await configureSidecarEngine(sidecarUrl, benchConfig);
  return { managed: true, sidecarUrl, child };
}

async function stopChildProcess(child) {
  if (!child || child.exitCode != null || child.killed) {
    return;
  }
  child.kill("SIGTERM");
  await new Promise((resolve) => setTimeout(resolve, 2000));
  if (child.exitCode == null && !child.killed) {
    child.kill("SIGKILL");
  }
}

export async function teardownManagedSidecar(handle) {
  if (!handle) {
    return;
  }

  if (shouldReleaseEngine()) {
    const result = await releaseSidecarEngine(handle.sidecarUrl);
    if (result?.released) {
      console.log("[sidecar] HF engine released; GPU memory should be freed");
    }
  }

  if (handle.managed && handle.child) {
    console.log("[sidecar] stopping managed sidecar process");
    await stopChildProcess(handle.child);
  }
}
