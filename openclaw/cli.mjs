#!/usr/bin/env node
/**
 * KVCOMM OpenClaw integration CLI.
 *
 *   node cli.mjs bench run [options]     — fixed-N sessions_spawn + TTFT benchmark
 *   node cli.mjs sidecar start           — start KVCOMM sidecar proxy
 *   node cli.mjs setup [dense|sidecar|clawbench-capability-sidecar] — apply openclaw.json profile
 *   node cli.mjs preflight               — check Gateway / vLLM / sidecar
 */

import { spawn, spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { OPENCLAW_MODULE_ROOT, BENCH_ROOT, scriptPath } from "./lib/paths.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

function printHelp() {
  console.log(`KVCOMM OpenClaw integration (kvcomm/openclaw)

Usage:
  node cli.mjs bench run [bench options...]   Run multi-agent TTFT benchmark
  node cli.mjs bench run-clawbench [opts]     ClawBench capability chain (workspace + tools)
  node cli.mjs sidecar start                  Start sidecar (Python FastAPI proxy)
  node cli.mjs setup [dense|sidecar]          Apply OpenClaw config profile
  node cli.mjs preflight                      Preflight checks

Bench options (passed to experiments/bench driver):
  --agent-count <n>           Override scenario agent_count (default: scenario file)
  --inference-mode <mode>     dense_prefill | kv_reuse
  --inference-backend <be>    vllm_direct | kvcomm_sidecar
  --warmup-runs <n>           Warmup runs (excluded from summary; for kv_reuse anchors)
  --measure-runs <n>          Measured runs (default: --runs or 1)
  --runs <n>                  Alias for --measure-runs
  --debug                     Debug mode (agent/stream timeout 60s unless overridden)
  --run-timeout-seconds <n>   Per-agent OpenClaw run + stream wait timeout
  --dry-run, --scenario, --dataset, --model, --task-id, --task-profile, --output, ...

Debug timeout (60s): BENCH_DEBUG=1, KVCOMM_BENCH_DEBUG=1, LOGURU_LEVEL=DEBUG, or --debug

Examples:
  # Dense baseline (OpenClaw → vLLM)
  node cli.mjs setup dense
  openclaw gateway run
  # ClawBench text chain (Phase 1)
  node cli.mjs bench run --agent-count 3 --measure-runs 3 --task-profile clawbench \\
    --dataset /src/KVCOMM/experiments/bench/datasets/tier1_clawbench.jsonl \\
    --task-id t1-fs-quick-note --model vllm/Qwen3-32B

  # ClawBench capability chain (workspace + tools + scoring)
  node cli.mjs setup clawbench-capability
  openclaw gateway run
  node cli.mjs bench run-clawbench --task-id t1-fs-quick-note --measure-runs 3 \\
    --model vllm/Qwen3-32B

  # Sidecar kvreuse path (bench auto-starts sidecar, loads GPU on first request, releases after run)
  node cli.mjs setup dual
  openclaw gateway run
  KVCOMM_HF_DEVICE=2,3,4 node cli.mjs bench run --inference-mode kv_reuse --inference-backend kvcomm_sidecar \\
    --warmup-runs 2 --measure-runs 3 --model kvcomm/Qwen3-32B --task-id micro-001

  # Optional: long-lived sidecar without GPU (stub proxy only)
  node cli.mjs sidecar start

  # ClawBench capability + sidecar kv_reuse
  node cli.mjs bench run-clawbench --inference-mode kv_reuse --inference-backend kvcomm_sidecar \\
    --warmup-runs 2 --measure-runs 3 --model kvcomm/Qwen3-32B --task-id t1-fs-quick-note
`);
}

function runScript(scriptName, args = []) {
  const script = scriptPath(scriptName);
  const result = spawnSync("bash", [script, ...args], { stdio: "inherit" });
  process.exit(result.status ?? 1);
}

function runBenchDriver(driverName, args) {
  const driver = join(BENCH_ROOT, `drivers/${driverName}`);
  const child = spawn(process.execPath, [driver, ...args], {
    stdio: "inherit",
    env: process.env,
    cwd: BENCH_ROOT,
  });
  child.on("exit", (code) => process.exit(code ?? 1));
}

function runBench(args) {
  runBenchDriver("run-o0-pre-chain.mjs", args);
}

function runClawbenchChain(args) {
  runBenchDriver("run-clawbench-chain.mjs", args);
}

function resolveSidecarPython() {
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

function runSidecar() {
  const server = join(OPENCLAW_MODULE_ROOT, "sidecar/server.py");
  const repoRoot = join(OPENCLAW_MODULE_ROOT, "..");
  const python = resolveSidecarPython();
  const hfDevice =
    process.env.KVCOMM_CUDA_VISIBLE_DEVICES?.trim() ||
    process.env.KVCOMM_HF_DEVICE?.trim() ||
    "";
  const env = {
    ...process.env,
    PYTHONPATH: [repoRoot, OPENCLAW_MODULE_ROOT, process.env.PYTHONPATH].filter(Boolean).join(":"),
  };
  if (hfDevice && !process.env.CUDA_VISIBLE_DEVICES?.trim()) {
    const parts = hfDevice.split(",").map((part) => part.trim()).filter(Boolean);
    env.CUDA_VISIBLE_DEVICES = parts.join(",");
    env.KVCOMM_HF_DEVICE = parts.map((_, index) => String(index)).join(",");
  }
  if (!env.KVCOMM_HF_MODEL?.trim() && !env.KVCOMM_HF_MODEL_PATH?.trim()) {
    console.log(
      "[sidecar] KVCOMM_HF_MODEL not set — running lightweight proxy (no GPU). " +
        "Bench with --inference-backend kvcomm_sidecar auto-starts HF engine when needed.",
    );
  } else if (!env.KVCOMM_DENSE_VIA_HF?.trim()) {
    env.KVCOMM_DENSE_VIA_HF = "1";
  }
  const child = spawn(python, [server], {
    stdio: "inherit",
    env,
    cwd: OPENCLAW_MODULE_ROOT,
  });
  child.on("exit", (code) => process.exit(code ?? 1));
}

function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h") {
    printHelp();
    return;
  }

  const [cmd, sub, ...rest] = argv;

  if (cmd === "bench" && sub === "run") {
    runBench(rest);
    return;
  }
  if (cmd === "bench" && sub === "run-clawbench") {
    runClawbenchChain(rest);
    return;
  }
  if (cmd === "sidecar" && sub === "start") {
    runSidecar();
    return;
  }
  if (cmd === "setup") {
    runScript("setup-openclaw.sh", rest.length ? rest : ["dual"]);
    return;
  }
  if (cmd === "preflight") {
    runScript("preflight.sh", rest);
    return;
  }

  console.error(`Unknown command: ${cmd} ${sub ?? ""}`.trim());
  printHelp();
  process.exit(1);
}

main();
