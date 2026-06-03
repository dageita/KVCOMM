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

import { appendJsonl, loadJson, loadJsonl } from "../lib/load-jsonl.mjs";
import { assertBenchGatewayConfig } from "../lib/openclaw-config.mjs";
import { renderTemplate, renderTemplateStrict } from "../lib/template.mjs";
import { connectGateway, runChainStackSpawn } from "../lib/spawn-stack.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    scenario: join(BENCH_ROOT, "scenarios/3agent-chain.json"),
    dataset: join(BENCH_ROOT, "datasets/tier0_copy.jsonl"),
    rolePrompt: join(BENCH_ROOT, "prompts/copy_machine.role.txt"),
    outputDir: join(BENCH_ROOT, "results"),
    experimentId: process.env.BENCH_EXPERIMENT_ID || "O0-pre-A",
    agentId: process.env.BENCH_AGENT_ID || "main",
    model: process.env.BENCH_MODEL || "",
    runs: 1,
    taskId: null,
    dryRun: false,
    negativeControl: null,
    runTimeoutSeconds: Number(process.env.BENCH_RUN_TIMEOUT_SECONDS || "600"),
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
    if (arg === "--task-id" && argv[i + 1]) {
      args.taskId = argv[++i];
      continue;
    }
    if (arg === "--negative-control" && argv[i + 1]) {
      args.negativeControl = argv[++i];
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function printHelp() {
  console.log(`Usage: node drivers/run-o0-pre-chain.mjs [options]

Options:
  --dry-run                 Validate dataset/scenario rendering only (no Gateway)
  --scenario <path>         Scenario JSON (default: scenarios/3agent-chain.json)
  --dataset <path>          Task JSONL (default: datasets/tier0_copy.jsonl)
  --output-dir <path>       Results directory (default: results/)
  --experiment-id <id>      Experiment label (default: O0-pre-A)
  --agent-id <id>           OpenClaw agent id (default: main)
  --model <provider/model>  Model override for subagent spawns
  --runs <n>                Repetitions per task (default: 1)
  --task-id <id>            Run a single task from dataset
  --negative-control NC-1   Drop agent_0 output from agent_1 task (L3 negative control)

Environment:
  OPENCLAW_GATEWAY_URL      ws://127.0.0.1:18789
  OPENCLAW_GATEWAY_TOKEN      Gateway auth token
  OPENCLAW_DIAGNOSTICS=timeline
  BENCH_EXPERIMENT_ID, BENCH_AGENT_ID, BENCH_MODEL
`);
}

function percentile(values, p) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

async function buildCopyRole(rolePromptPath) {
  const raw = await readFile(rolePromptPath, "utf8");
  const prefixRepeats = Number(process.env.COPY_PREFIX_REPEATS || "64");
  const outLength = Number(process.env.COPY_OUT_LENGTH || "128");
  const prefix = " Ω".repeat(Math.max(0, prefixRepeats)).trimStart();
  const roleBody = renderTemplateStrict(raw.trim(), { out_length: String(outLength) });
  return `${prefix}\n${roleBody}`.trim();
}

async function prepareTasks(datasetPath, taskId, copyRole) {
  const rows = await loadJsonl(datasetPath);
  const filtered = taskId ? rows.filter((row) => row.task_id === taskId) : rows;
  if (filtered.length === 0) {
    throw new Error(`No tasks found in ${datasetPath}${taskId ? ` for task-id ${taskId}` : ""}`);
  }
  return filtered.map((row) => ({
    ...row,
    agent_tasks: Object.fromEntries(
      Object.entries(row.agent_tasks).map(([key, template]) => [
        key,
        renderTemplate(template, { copy_role: copyRole }),
      ]),
    ),
  }));
}

async function dryRunValidate(scenario, tasks) {
  console.log("[dry-run] scenario:", scenario.id, "topology:", scenario.topology);
  for (const task of tasks) {
    const outputs = { user_question: task.user_question };
    for (let i = 0; i < scenario.agent_count; i += 1) {
      const key = `agent_${i}`;
      const text = renderTemplate(task.agent_tasks[key], outputs);
      outputs[`agent_${i}_current`] = `<mock-output-${i}>`;
      console.log(`[dry-run] ${task.task_id} ${key} chars=${text.length} ok=${text.length > 0}`);
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
  const probe = rows.filter((row) => row.probe && typeof row.ttft_ms === "number");
  const ttfts = probe.map((row) => row.ttft_ms);
  const summary = {
    rows: rows.length,
    probe_rows: probe.length,
    ttft_p50: percentile(ttfts, 50),
    ttft_p99: percentile(ttfts, 99),
    ttft_fallback_rate:
      probe.length === 0
        ? 1
        : probe.filter((row) => row.ttft_fallback).length / probe.length,
    comms_ok_rate:
      rows.length === 0
        ? 0
        : rows.filter((row) => row.task_includes_upstream !== false).length / rows.length,
  };
  return summary;
}

async function main() {
  const args = parseArgs(process.argv);
  const scenario = await loadJson(args.scenario);
  const copyRole = await buildCopyRole(args.rolePrompt);
  const tasks = await prepareTasks(args.dataset, args.taskId, copyRole);

  if (args.dryRun) {
    await dryRunValidate(scenario, tasks);
    console.log("[dry-run] OK");
    return;
  }

  if (args.model && (args.model.startsWith("/") || args.model.includes("\\"))) {
    console.warn(
      `[bench] --model looks like a filesystem path (${args.model}). ` +
        "OpenClaw expects a configured provider ref, e.g. vllm/meta-llama/Llama-3.1-8B-Instruct",
    );
  }

  await mkdir(args.outputDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputPath = join(args.outputDir, `${args.experimentId}_${stamp}.jsonl`);

  await assertBenchGatewayConfig();

  console.log(`[bench] connecting gateway experiment=${args.experimentId} tasks=${tasks.length}`);
  const client = await connectGateway();

  try {
    for (let runIndex = 0; runIndex < args.runs; runIndex += 1) {
      for (const taskRow of tasks) {
        const runId = randomUUID();
        const label = `kvcomm-bench-${taskRow.task_id}-run${runIndex}-${runId.slice(0, 8)}`;
        const orchestratorSessionKey = await client.createSession({
          agentId: args.agentId,
          model: args.model || undefined,
          label,
        });

        console.log(
          `[bench] run=${runIndex + 1}/${args.runs} task=${taskRow.task_id} session=${orchestratorSessionKey}`,
        );

        const result = await runChainStackSpawn(client, {
          orchestratorSessionKey,
          scenario,
          taskRow,
          model: args.model || undefined,
          runTimeoutSeconds: args.runTimeoutSeconds,
          experimentId: args.experimentId,
          negativeControl: args.negativeControl,
          runId,
        });

        for (const record of result.records) {
          await appendJsonl(outputPath, record);
        }
        await appendJsonl(outputPath, {
          type: "run_summary",
          experiment_id: args.experimentId,
          task_id: taskRow.task_id,
          run_id: runId,
          e2e_run_ms: result.e2e_run_ms,
          orchestrator_session_key: orchestratorSessionKey,
          timestamp: new Date().toISOString(),
        });

        console.log(
          `[bench] done task=${taskRow.task_id} e2e_ms=${result.e2e_run_ms} probe_ttft=${result.records.find((r) => r.probe)?.ttft_ms ?? "n/a"}`,
        );
      }
    }
  } finally {
    await client.close();
  }

  const summary = await summarizeResults(outputPath);
  const summaryPath = outputPath.replace(/\.jsonl$/, ".summary.json");
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");

  console.log("[bench] results:", outputPath);
  console.log("[bench] summary:", summaryPath, summary);
}

main().catch((err) => {
  console.error("[bench] fatal:", err);
  process.exit(1);
});
