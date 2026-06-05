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
import { resolveTaskBody } from "../lib/kvcomm-task.mjs";
import { assertBenchGatewayConfig } from "../lib/openclaw-config.mjs";
import { renderTemplate, renderTemplateStrict } from "../lib/template.mjs";
import { connectGateway, runChainStackSpawn } from "../lib/spawn-stack.mjs";
import { summarizeBenchRows } from "../lib/summarize-results.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    scenario: join(BENCH_ROOT, "scenarios/3agent-chain.json"),
    dataset: join(BENCH_ROOT, "datasets/tier0_copy.jsonl"),
    rolePrompt: join(BENCH_ROOT, "prompts/copy_machine.role.txt"),
    outputDir: join(BENCH_ROOT, "results"),
    output: process.env.BENCH_OUTPUT?.trim() || null,
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
  --runs <n>                Repetitions per task (default: 1)
  --task-id <id>            Run a single task from dataset
  --negative-control NC-1   Drop agent_0 output from agent_1 task (L3 negative control)

Environment:
  OPENCLAW_GATEWAY_URL      ws://127.0.0.1:18789
  OPENCLAW_GATEWAY_TOKEN      Gateway auth token
  OPENCLAW_DIAGNOSTICS=timeline
  BENCH_EXPERIMENT_ID, BENCH_AGENT_ID, BENCH_MODEL, BENCH_OUTPUT
`);
}

async function buildCopyRole(rolePromptPath) {
  const raw = await readFile(rolePromptPath, "utf8");
  const prefixRepeats = Number(process.env.COPY_PREFIX_REPEATS || "64");
  const outLength = Number(process.env.COPY_OUT_LENGTH || "128");
  const prefix = " Ω".repeat(Math.max(0, prefixRepeats));
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

async function main() {
  const args = parseArgs(process.argv);
  const scenario = await loadJson(args.scenario);
  const copyRole = await buildCopyRole(args.rolePrompt);
  const tasks = await prepareTasks(args.dataset, args.taskId, copyRole);

  if (args.dryRun) {
    await dryRunValidate(scenario, tasks, 0);
    console.log("[dry-run] OK (task_body from kvcomm_task fixture / task_body field)");
    return;
  }

  if (args.model && (args.model.startsWith("/") || args.model.includes("\\"))) {
    console.warn(
      `[bench] --model looks like a filesystem path (${args.model}). ` +
        "OpenClaw expects a configured provider ref, e.g. vllm/meta-llama/Llama-3.1-8B-Instruct",
    );
  }

  await mkdir(args.outputDir, { recursive: true });
  const { jsonl: outputPath, summary: summaryPath } = resolveOutputPaths({
    outputDir: args.outputDir,
    experimentId: args.experimentId,
    output: args.output,
  });
  await mkdir(dirname(outputPath), { recursive: true });

  await assertBenchGatewayConfig();

  console.log(
    `[bench] connecting gateway experiment=${args.experimentId} tasks=${tasks.length} output=${outputPath}`,
  );
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

        const taskBody = await resolveTaskBody(taskRow, runIndex);
        const result = await runChainStackSpawn(client, {
          orchestratorSessionKey,
          scenario,
          taskRow: { ...taskRow, task_body: taskBody },
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
  await writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");

  console.log("[bench] results:", outputPath);
  console.log("[bench] summary:", summaryPath, summary);
}

main().catch((err) => {
  console.error("[bench] fatal:", err);
  process.exit(1);
});
