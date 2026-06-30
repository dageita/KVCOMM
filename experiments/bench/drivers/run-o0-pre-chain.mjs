#!/usr/bin/env node
/**
 * O0-pre spike: OpenClaw subagent stack (Chain 3-agent) + TTFT collector.
 *
 * Prerequisites:
 *   - OpenClaw Gateway running with diagnostics.timeline enabled:
 *       OPENCLAW_DIAGNOSTICS=timeline openclaw gateway run
 *   - OPENCLAW_GATEWAY_TOKEN set if gateway requires auth
 *   - Orchestrator agent (default: main) with sessions_spawn available (coding/full profile)
 *
 * Usage:
 *   cd KVCOMM/experiments/bench && npm install
 *   npm run dry-run
 *   npm run run -- --runs 1 --task-id micro-001
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

import { appendJsonl, initJsonlOutput, loadJson, loadJsonl } from "../lib/load-jsonl.mjs";
import { resolveTaskBody } from "../lib/kvcomm-task.mjs";
import { assertBenchGatewayConfig } from "../lib/openclaw-config.mjs";
import { renderTemplate, renderTemplateStrict } from "../lib/template.mjs";
import { connectGateway, runChainStackSpawn } from "../lib/spawn-stack.mjs";
import { summarizeBenchRows } from "../lib/summarize-results.mjs";
import { writeBenchReport } from "../lib/bench-report.mjs";
import {
  buildRunMetadata,
  enrichAgentRecord,
  enrichRunSummary,
} from "../../../openclaw/lib/bench-metadata.mjs";
import {
  buildScenario,
  extendTaskAgentTemplates,
} from "../../../openclaw/lib/scenario-factory.mjs";
import { OPENCLAW_MODULE_ROOT } from "../../../openclaw/lib/paths.mjs";
import {
  ensureManagedSidecarForBench,
  teardownManagedSidecar,
} from "../../../openclaw/lib/sidecar-lifecycle.mjs";
import { spawnSync } from "node:child_process";
import { isBenchDebugMode, resolveRunTimeoutSeconds } from "../lib/bench-timeout.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    scenario: join(BENCH_ROOT, "scenarios/3agent-chain.json"),
    dataset: join(BENCH_ROOT, "datasets/tier0_copy.jsonl"),
    rolePrompt: join(BENCH_ROOT, "prompts/copy_machine.role.txt"),
    taskProfile: "copy",
    outputDir: join(BENCH_ROOT, "results"),
    output: process.env.BENCH_OUTPUT?.trim() || null,
    experimentId: process.env.BENCH_EXPERIMENT_ID || "O0-pre-A",
    agentId: process.env.BENCH_AGENT_ID || "main",
    model: process.env.BENCH_MODEL || "",
    runs: 1,
    measureRuns: null,
    warmupRuns: 0,
    agentCount: null,
    inferenceMode: process.env.KVCOMM_MODE?.trim() || "dense_prefill",
    inferenceBackend: process.env.KVCOMM_INFERENCE_BACKEND?.trim() || null,
    taskId: null,
    dryRun: false,
    debug: false,
    runTimeoutSecondsExplicit: null,
    runTimeoutSeconds: 600,
    negativeControl: null,
    cleanSessions: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--dry-run") {
      args.dryRun = true;
      continue;
    }
    if (arg === "--scenario" && argv[i + 1]) {
      args.scenario = resolve(argv[++i]);
      continue;
    }
    if (arg === "--dataset" && argv[i + 1]) {
      args.dataset = resolve(argv[++i]);
      continue;
    }
    if (arg === "--output-dir" && argv[i + 1]) {
      args.outputDir = resolve(argv[++i]);
      continue;
    }
    if (arg === "--output" && argv[i + 1]) {
      args.output = argv[++i].trim();
      continue;
    }
    if (arg === "--experiment-id" && argv[i + 1]) {
      args.experimentId = argv[++i];
      continue;
    }
    if (arg === "--agent-id" && argv[i + 1]) {
      args.agentId = argv[++i];
      continue;
    }
    if (arg === "--model" && argv[i + 1]) {
      args.model = argv[++i];
      continue;
    }
    if (arg === "--runs" && argv[i + 1]) {
      args.runs = Math.max(1, Number(argv[++i]) || 1);
      continue;
    }
    if (arg === "--measure-runs" && argv[i + 1]) {
      args.measureRuns = Math.max(1, Number(argv[++i]) || 1);
      continue;
    }
    if (arg === "--warmup-runs" && argv[i + 1]) {
      args.warmupRuns = Math.max(0, Number(argv[++i]) || 0);
      continue;
    }
    if (arg === "--agent-count" && argv[i + 1]) {
      args.agentCount = Math.max(1, Number(argv[++i]) || 1);
      continue;
    }
    if (arg === "--inference-mode" && argv[i + 1]) {
      args.inferenceMode = argv[++i];
      continue;
    }
    if (arg === "--inference-backend" && argv[i + 1]) {
      args.inferenceBackend = argv[++i];
      continue;
    }
    if (arg === "--clean-sessions") {
      args.cleanSessions = true;
      continue;
    }
    if (arg === "--task-id" && argv[i + 1]) {
      args.taskId = argv[++i];
      continue;
    }
    if (arg === "--task-profile" && argv[i + 1]) {
      args.taskProfile = argv[++i];
      continue;
    }
    if (arg === "--role-prompt" && argv[i + 1]) {
      args.rolePrompt = resolve(argv[++i]);
      continue;
    }
    if (arg === "--negative-control" && argv[i + 1]) {
      args.negativeControl = argv[++i];
      continue;
    }
    if (arg === "--debug") {
      args.debug = true;
      continue;
    }
    if (arg === "--run-timeout-seconds" && argv[i + 1]) {
      args.runTimeoutSecondsExplicit = Number(argv[++i]);
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  args.runTimeoutSeconds = resolveRunTimeoutSeconds({
    explicitSeconds: args.runTimeoutSecondsExplicit,
    debugFlag: args.debug,
  });
  return args;
}

/**
 * Resolve jsonl + summary paths. --output is a basename (no dir) under output-dir,
 * or a path ending in .jsonl (summary sits beside it).
 */
function resolveOutputPaths({ outputDir, experimentId, output }) {
  if (!output) {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const base = `${experimentId}_${stamp}`;
    return {
      jsonl: join(outputDir, `${base}.jsonl`),
      summary: join(outputDir, `${base}.summary.json`),
    };
  }

  let name = output;
  if (name.endsWith(".summary.json")) {
    name = name.slice(0, -".summary.json".length);
  } else if (name.endsWith(".jsonl")) {
    name = name.slice(0, -".jsonl".length);
  }

  const hasPathSep = /[\\/]/.test(name);
  if (hasPathSep) {
    const jsonl = resolve(`${name}.jsonl`);
    return {
      jsonl,
      summary: jsonl.replace(/\.jsonl$/i, ".summary.json"),
    };
  }

  return {
    jsonl: join(outputDir, `${name}.jsonl`),
    summary: join(outputDir, `${name}.summary.json`),
  };
}

function printHelp() {
  console.log(`Usage: node drivers/run-o0-pre-chain.mjs [options]

Options:
  --dry-run                 Validate dataset/scenario rendering only (no Gateway)
  --scenario <path>         Scenario JSON (default: scenarios/3agent-chain.json)
  --dataset <path>          Task JSONL (default: datasets/tier0_copy.jsonl)
  --output-dir <path>       Results directory (default: results/)
  --output <name>           Result basename or path (default: <experiment-id>_<timestamp>)
  --experiment-id <id>      Experiment label in jsonl rows (default: O0-pre-A)
  --agent-id <id>           OpenClaw agent id (default: main)
  --model <provider/model>  Model override for subagent spawns
  --runs <n>                Measured repetitions per task (alias for --measure-runs)
  --measure-runs <n>        Measured runs included in summary (default: --runs or 1)
  --warmup-runs <n>         Warmup runs excluded from summary (kv_reuse anchor init)
  --agent-count <n>         Override scenario agent_count (dynamic Chain templates)
  --inference-mode <mode>   dense_prefill | kv_reuse
  --inference-backend <be>  vllm_direct | kvcomm_sidecar
  --clean-sessions          Purge main agent session store before run (Gateway should be stopped)
  --task-id <id>            Run a single task from dataset
  --task-profile <profile>  copy | clawbench (default: copy)
  --role-prompt <path>      Role prompt template (default depends on task-profile)
  --negative-control NC-1   Drop agent_0 output from agent_1 task (L3 negative control)
  --debug                   Debug mode (agent/stream timeout 60s unless overridden)
  --run-timeout-seconds <n> Per-agent OpenClaw run + stream wait timeout (default: 600, debug: 60)

Environment:
  OPENCLAW_GATEWAY_URL      ws://127.0.0.1:18789
  OPENCLAW_GATEWAY_TOKEN      Gateway auth token
  OPENCLAW_DIAGNOSTICS=timeline
  BENCH_EXPERIMENT_ID, BENCH_AGENT_ID, BENCH_MODEL, BENCH_OUTPUT
  KVCOMM_MODE, KVCOMM_INFERENCE_BACKEND, KVCOMM_SIDECAR_URL
  BENCH_DEBUG=1 | KVCOMM_BENCH_DEBUG=1 | LOGURU_LEVEL=DEBUG  → 60s timeout
  BENCH_RUN_TIMEOUT_SECONDS=<n>    Override timeout explicitly
`);
}

async function buildRolePrompt(rolePromptPath, taskProfile) {
  const raw = await readFile(rolePromptPath, "utf8");
  if (taskProfile === "clawbench") {
    return raw.trim();
  }
  const prefixRepeats = Number(process.env.COPY_PREFIX_REPEATS || "64");
  const outLength = Number(process.env.COPY_OUT_LENGTH || "128");
  const prefix = " Ω".repeat(Math.max(0, prefixRepeats));
  const roleBody = renderTemplateStrict(raw.trim(), { out_length: String(outLength) });
  return `${prefix}\n${roleBody}`.trim();
}

async function prepareTasks(datasetPath, taskId, rolePrompt, agentCount, topology, taskProfile) {
  const rows = await loadJsonl(datasetPath);
  const filtered = taskId ? rows.filter((row) => row.task_id === taskId) : rows;
  if (filtered.length === 0) {
    throw new Error(`No tasks found in ${datasetPath}${taskId ? ` for task-id ${taskId}` : ""}`);
  }

  const templateVars =
    taskProfile === "clawbench"
      ? { role_prompt: rolePrompt }
      : { copy_role: rolePrompt };

  return filtered.map((row) => {
    const extended =
      agentCount != null ? extendTaskAgentTemplates(row, agentCount, topology) : row;
    return {
      ...extended,
      agent_tasks: Object.fromEntries(
        Object.entries(extended.agent_tasks).map(([key, template]) => [
          key,
          renderTemplate(template, templateVars),
        ]),
      ),
      ...(extended.capability_agent_tasks
        ? {
            capability_agent_tasks: Object.fromEntries(
              Object.entries(extended.capability_agent_tasks).map(([key, template]) => [
                key,
                renderTemplate(template, templateVars),
              ]),
            ),
          }
        : {}),
    };
  });
}

async function dryRunValidate(scenario, tasks, runIndex = 0) {
  console.log("[dry-run] scenario:", scenario.id, "topology:", scenario.topology);
  for (const task of tasks) {
    const taskBody = await resolveTaskBody(task, runIndex);
    const outputs = { task_body: taskBody };
    for (let i = 0; i < scenario.agent_count; i += 1) {
      const key = `agent_${i}`;
      const text = renderTemplate(task.agent_tasks[key], outputs);
      outputs[`agent_${i}_current`] = `<mock-output-${i}>`;
      console.log(
        `[dry-run] ${task.task_id} run=${runIndex} ${key} chars=${text.length} task_chars=${taskBody.length}`,
      );
    }
  }
}

async function summarizeResults(outputPath) {
  const raw = await readFile(outputPath, "utf8");
  const rows = raw
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return summarizeBenchRows(rows);
}

async function maybeCleanBenchSessions(enabled) {
  if (!enabled) {
    return;
  }
  const script = join(OPENCLAW_MODULE_ROOT, "scripts/clean-bench-sessions.sh");
  console.log("[bench] cleaning main agent sessions before run...");
  const result = spawnSync("bash", [script], { stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error(`clean-bench-sessions.sh failed with exit ${result.status ?? "unknown"}`);
  }
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.debug || isBenchDebugMode()) {
    console.log(`[bench] debug mode: agent_run_timeout_seconds=${args.runTimeoutSeconds}`);
  }
  if (!["copy", "clawbench"].includes(args.taskProfile)) {
    throw new Error(`Unknown task profile: ${args.taskProfile} (expected copy or clawbench)`);
  }
  if (args.taskProfile === "clawbench" && args.rolePrompt.endsWith("copy_machine.role.txt")) {
    args.rolePrompt = join(BENCH_ROOT, "prompts/clawbench_chain.role.minimal.txt");
  }

  const measureRuns = args.measureRuns ?? args.runs;
  const baseScenario = await loadJson(args.scenario);
  const scenario =
    args.agentCount != null
      ? buildScenario(baseScenario, args.agentCount, baseScenario.topology ?? "chain")
      : baseScenario;
  const rolePrompt = await buildRolePrompt(args.rolePrompt, args.taskProfile);
  const tasks = await prepareTasks(
    args.dataset,
    args.taskId,
    rolePrompt,
    scenario.agent_count,
    scenario.topology ?? "chain",
    args.taskProfile,
  );

  const runMetadataBase = buildRunMetadata({
    experimentId: args.experimentId,
    scenario,
    model: args.model,
    inferenceMode: args.inferenceMode,
    inferenceBackend: args.inferenceBackend,
    taskProfile: args.taskProfile,
  });

  if (args.dryRun) {
    await dryRunValidate(scenario, tasks, 0);
    console.log("[dry-run] OK (task_body from kvcomm_task fixture / task_body field)");
    console.log("[dry-run] metadata:", runMetadataBase);
    return;
  }

  if (args.model && (args.model.startsWith("/") || args.model.includes("\\"))) {
    console.warn(
      `[bench] --model looks like a filesystem path (${args.model}). ` +
        "OpenClaw expects a configured provider ref, e.g. vllm/Qwen3-32B",
    );
  }

  await mkdir(args.outputDir, { recursive: true });
  const { jsonl: outputPath, summary: summaryPath } = resolveOutputPaths({
    outputDir: args.outputDir,
    experimentId: args.experimentId,
    output: args.output,
  });
  await mkdir(dirname(outputPath), { recursive: true });
  await initJsonlOutput(outputPath);

  await maybeCleanBenchSessions(args.cleanSessions);
  await assertBenchGatewayConfig();

  console.log(
    `[bench] connecting gateway experiment=${args.experimentId} tasks=${tasks.length} ` +
      `agents=${scenario.agent_count} mode=${runMetadataBase.inference_mode} ` +
      `backend=${runMetadataBase.inference_backend} output=${outputPath}`,
  );
  const client = await connectGateway();
  const sidecarHandle = await ensureManagedSidecarForBench({
    inferenceBackend: args.inferenceBackend ?? runMetadataBase.inference_backend,
  });

  const totalRuns = args.warmupRuns + measureRuns;

  try {
    for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
      const isWarmup = runIndex < args.warmupRuns;
      const runMetadata = buildRunMetadata({
        experimentId: args.experimentId,
        scenario,
        model: args.model,
        inferenceMode: args.inferenceMode,
        inferenceBackend: args.inferenceBackend,
        taskProfile: args.taskProfile,
        warmup: isWarmup,
      });

      for (const taskRow of tasks) {
        const runId = randomUUID();
        const runUid = runId.slice(0, 8);

        const phase = isWarmup ? "warmup" : "measure";
        console.log(
          `[bench] ${phase}=${runIndex + 1}/${totalRuns} task=${taskRow.task_id} run_uid=${runUid}`,
        );

        const taskBody = await resolveTaskBody(taskRow, runIndex);
        const effectiveInferenceMode = isWarmup ? "dense_prefill" : args.inferenceMode;
        const result = await runChainStackSpawn(client, {
          agentId: args.agentId,
          scenario,
          taskRow: { ...taskRow, task_body: taskBody },
          model: args.model || undefined,
          runTimeoutSeconds: args.runTimeoutSeconds,
          experimentId: args.experimentId,
          negativeControl: args.negativeControl,
          runId,
          inferenceMode: effectiveInferenceMode,
          inferenceBackend: args.inferenceBackend ?? runMetadata.inference_backend,
          taskProfile: args.taskProfile,
        });

        for (const record of result.records) {
          await appendJsonl(outputPath, enrichAgentRecord(record, runMetadata, runUid));
        }
        await appendJsonl(
          outputPath,
          enrichRunSummary(
            {
              type: "run_summary",
              task_id: taskRow.task_id,
              run_id: runId,
              e2e_run_ms: result.e2e_run_ms,
              orchestrator_session_keys: result.orchestrator_session_keys,
              timestamp: new Date().toISOString(),
            },
            runMetadata,
            runUid,
          ),
        );

        if (!isWarmup) {
          console.log(
            `[bench] done task=${taskRow.task_id} e2e_ms=${result.e2e_run_ms} probe_ttft=${result.records.find((r) => r.probe)?.ttft_ms ?? "n/a"}`,
          );
        } else {
          console.log(`[bench] warmup done task=${taskRow.task_id} e2e_ms=${result.e2e_run_ms}`);
        }
      }
    }
  } finally {
    await client.close();
    await teardownManagedSidecar(sidecarHandle);
  }

  if (measureRuns === 0) {
    console.log("[bench] warmup-only run; no summary written");
    return;
  }

  const summary = await summarizeResults(outputPath);
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  const reportPath = await writeBenchReport({
    outputPath,
    rows: (await readFile(outputPath, "utf8"))
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line)),
    summary,
  });

  console.log("[bench] results:", outputPath);
  console.log("[bench] summary:", summaryPath, summary);
  console.log("[bench] report:", reportPath);
}

main().catch((err) => {
  console.error("[bench] fatal:", err);
  process.exit(1);
});
