#!/usr/bin/env node
/**
 * ClawBench capability lane: fixed Chain N-agent with shared workspace + tools + scoring.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";

import { appendJsonl, initJsonlOutput, loadJson, loadJsonl } from "../lib/load-jsonl.mjs";
import { resolveTaskBody } from "../lib/kvcomm-task.mjs";
import { assertBenchGatewayConfig, assertKvcommSidecarGatewayModel } from "../lib/openclaw-config.mjs";
import { renderTemplate, renderTemplateStrict } from "../lib/template.mjs";
import { connectGateway, runChainStackSpawn } from "../lib/spawn-stack.mjs";
import { summarizeBenchRows, summarizeClawbenchCapability } from "../lib/summarize-results.mjs";
import { writeBenchReport } from "../lib/bench-report.mjs";
import {
  buildChainTranscript,
  collectSessionMessages,
  scoreCapabilityRun,
  slimCapabilityScore,
  setupClawbenchWorkspace,
  stageCapabilityWorkspaceForAgents,
  syncCapabilityWorkspaceArtifacts,
} from "../lib/clawbench-chain.mjs";
import {
  buildRunMetadata,
  computeRunKvReuseStats,
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
import { isBenchDebugMode, resolveRunTimeoutSeconds } from "../lib/bench-timeout.mjs";
import {
  resolveBenchPaddingBlock,
  resolveBenchPaddingEnabled,
  resolveRolePromptPath,
  resolveTaskBodyForBench,
} from "../lib/bench-padding.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    scenario: join(BENCH_ROOT, "scenarios/3agent-chain.json"),
    dataset: join(BENCH_ROOT, "datasets/tier1_clawbench.jsonl"),
    rolePrompt: join(BENCH_ROOT, "prompts/clawbench_chain.role.minimal.txt"),
    outputDir: join(BENCH_ROOT, "results"),
    output: process.env.BENCH_OUTPUT?.trim() || null,
    experimentId: process.env.BENCH_EXPERIMENT_ID || "clawbench-chain",
    agentId: process.env.BENCH_AGENT_ID || "main",
    model: process.env.BENCH_MODEL || "",
    runs: 1,
    measureRuns: null,
    warmupRuns: 0,
    agentCount: 3,
    inferenceMode: process.env.KVCOMM_MODE?.trim() || "dense_prefill",
    inferenceBackend: process.env.KVCOMM_INFERENCE_BACKEND?.trim() || null,
    taskId: null,
    dryRun: false,
    debug: false,
    runTimeoutSecondsExplicit: null,
    runTimeoutSeconds: 600,
    cleanSessions: false,
    judgeModel: process.env.CLAWBENCH_JUDGE_MODEL?.trim() || "",
    skipScore: false,
    benchPadding: resolveBenchPaddingEnabled(process.env.BENCH_PADDING),
    benchPaddingExplicit: process.env.BENCH_PADDING != null && process.env.BENCH_PADDING !== "",
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
    if (arg === "--role-prompt" && argv[i + 1]) {
      args.rolePrompt = resolve(argv[++i]);
      continue;
    }
    if (arg === "--judge-model" && argv[i + 1]) {
      args.judgeModel = argv[++i];
      continue;
    }
    if (arg === "--skip-score") {
      args.skipScore = true;
      continue;
    }
    if (arg === "--bench-padding" && argv[i + 1]) {
      args.benchPadding = resolveBenchPaddingEnabled(argv[++i]);
      args.benchPaddingExplicit = true;
      continue;
    }
    if (arg === "--no-bench-padding") {
      args.benchPadding = false;
      args.benchPaddingExplicit = true;
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
  if (!args.benchPaddingExplicit) {
    args.rolePrompt = resolveRolePromptPath(args.benchPadding);
  } else if (!args.benchPadding) {
    args.rolePrompt = resolveRolePromptPath(false);
  } else {
    args.rolePrompt = resolveRolePromptPath(true);
  }
  return args;
}

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
    return { jsonl, summary: jsonl.replace(/\.jsonl$/i, ".summary.json") };
  }
  return {
    jsonl: join(outputDir, `${name}.jsonl`),
    summary: join(outputDir, `${name}.summary.json`),
  };
}

function printHelp() {
  console.log(`Usage: node drivers/run-clawbench-chain.mjs [options]

ClawBench capability lane (shared workspace + tools + scoring).

Options:
  --dry-run
  --dataset <path>            default: datasets/tier1_clawbench.jsonl
  --task-id <id>
  --agent-count <n>           default: 3
  --measure-runs <n>
  --warmup-runs <n>
  --model <provider/model>
  --judge-model <provider/model>   Optional ClawBench judge model
  --skip-score                     Skip Python capability scoring
  --bench-padding on|off           Inject long stable KV padding into role + task prompts (default: off)
  --no-bench-padding               Same as --bench-padding off
  --debug                          Debug mode (agent/stream timeout 60s unless overridden)
  --run-timeout-seconds <n>        Per-agent OpenClaw run + stream wait timeout (default: 600, debug: 60)
  --output, --output-dir, --experiment-id, --clean-sessions

Environment:
  BENCH_DEBUG=1 | KVCOMM_BENCH_DEBUG=1 | LOGURU_LEVEL=DEBUG  → 60s timeout
  BENCH_RUN_TIMEOUT_SECONDS=<n>    Override timeout explicitly
  BENCH_PADDING=on|off             Same as --bench-padding (default: off)
`);
}

async function buildRolePrompt(rolePromptPath) {
  const raw = await readFile(rolePromptPath, "utf8");
  return raw.trim();
}

async function prepareTasks(datasetPath, taskId, rolePrompt, agentCount, topology, benchPadding) {
  const rows = await loadJsonl(datasetPath);
  const filtered = taskId ? rows.filter((row) => row.task_id === taskId) : rows;
  if (filtered.length === 0) {
    throw new Error(`No tasks found in ${datasetPath}${taskId ? ` for task-id ${taskId}` : ""}`);
  }
  const templateVars = { role_prompt: rolePrompt };
  const prepared = [];
  for (const row of filtered) {
    const benchPaddingBlock = await resolveBenchPaddingBlock(row, benchPadding);
    const vars = { ...templateVars, bench_padding: benchPaddingBlock };
    const extended = extendTaskAgentTemplates(row, agentCount, topology);
    const renderMap = (templates) =>
      Object.fromEntries(
        Object.entries(templates).map(([key, template]) => [key, renderTemplate(template, vars)]),
      );
    prepared.push({
      ...extended,
      agent_tasks: renderMap(extended.agent_tasks),
      ...(extended.capability_agent_tasks
        ? { capability_agent_tasks: renderMap(extended.capability_agent_tasks) }
        : {}),
      _bench_padding_enabled: benchPadding,
      _bench_role_prompt: rolePrompt,
    });
  }
  return prepared;
}

async function dryRunValidate(scenario, tasks, runIndex = 0) {
  console.log("[dry-run] clawbench capability scenario:", scenario.id);
  for (const task of tasks) {
    const taskBody = await resolveTaskBody(task, runIndex);
    const templates = task.capability_agent_tasks ?? task.agent_tasks;
    console.log(`[dry-run] ${task.task_id} asset_packs=${JSON.stringify(task.clawbench_ref?.asset_packs ?? [])}`);
    for (let i = 0; i < scenario.agent_count; i += 1) {
      const key = `agent_${i}`;
      const text = renderTemplate(templates[key], { task_body: taskBody, agent_0_current: "<mock>" });
      console.log(`[dry-run] ${task.task_id} ${key} chars=${text.length}`);
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
  const bench = summarizeBenchRows(rows);
  const clawbench = summarizeClawbenchCapability(rows);
  return clawbench ? { ...bench, clawbench_capability: clawbench } : bench;
}

async function maybeCleanBenchSessions(enabled) {
  if (!enabled) {
    return;
  }
  const script = join(OPENCLAW_MODULE_ROOT, "scripts/clean-bench-sessions.sh");
  const result = spawnSync("bash", [script], { stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error(`clean-bench-sessions.sh failed with exit ${result.status ?? "unknown"}`);
  }
}

async function main() {
  const args = parseArgs(process.argv);
  process.env.KVCOMM_BENCH_PADDING = args.benchPadding ? "1" : "0";
  process.env.BENCH_PADDING = args.benchPadding ? "on" : "off";
  const measureRuns = args.measureRuns ?? args.runs;
  if (args.debug || isBenchDebugMode()) {
    console.log(
      `[clawbench-chain] debug mode: agent_run_timeout_seconds=${args.runTimeoutSeconds}`,
    );
  }
  const baseScenario = await loadJson(args.scenario);
  const scenario = buildScenario(baseScenario, args.agentCount, baseScenario.topology ?? "chain");
  const rolePrompt = await buildRolePrompt(args.rolePrompt);
  const tasks = await prepareTasks(
    args.dataset,
    args.taskId,
    rolePrompt,
    scenario.agent_count,
    scenario.topology ?? "chain",
    args.benchPadding,
  );

  console.log(
    `[clawbench-chain] bench_padding=${args.benchPadding ? "on" : "off"} role=${args.rolePrompt}`,
  );

  const runMetadataBase = buildRunMetadata({
    experimentId: args.experimentId,
    scenario,
    model: args.model,
    inferenceMode: args.inferenceMode,
    inferenceBackend: args.inferenceBackend,
    taskProfile: "clawbench",
    spawnMode: "capability",
  });

  if (args.dryRun) {
    await dryRunValidate(scenario, tasks, 0);
    console.log("[dry-run] OK");
    console.log("[dry-run] metadata:", runMetadataBase);
    return;
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
  const { configPath, config } = await assertBenchGatewayConfig();
  if (args.inferenceBackend) {
    process.env.KVCOMM_INFERENCE_BACKEND = args.inferenceBackend;
  }
  assertKvcommSidecarGatewayModel(config, { configPath, model: args.model });

  console.log(
    `[clawbench-chain] connecting gateway tasks=${tasks.length} agents=${scenario.agent_count} output=${outputPath}`,
  );
  const client = await connectGateway();
  const sidecarHandle = await ensureManagedSidecarForBench({
    inferenceBackend: args.inferenceBackend ?? runMetadataBase.inference_backend,
  });
  const totalRuns = args.warmupRuns + measureRuns;
  let measureIndex = 0;

  try {
    for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
      const isWarmup = runIndex < args.warmupRuns;
      const runMetadata = buildRunMetadata({
        experimentId: args.experimentId,
        scenario,
        model: args.model,
        inferenceMode: args.inferenceMode,
        inferenceBackend: args.inferenceBackend,
        taskProfile: "clawbench",
        spawnMode: "capability",
        warmup: isWarmup,
      });

      for (const taskRow of tasks) {
        const runId = randomUUID();
        const runUid = runId.slice(0, 8);
        const assetPacks = taskRow.clawbench_ref?.asset_packs ?? [];
        if (assetPacks.length === 0) {
          throw new Error(`Task ${taskRow.task_id} missing clawbench_ref.asset_packs`);
        }

        const workspaceDir = setupClawbenchWorkspace({
          taskId: taskRow.task_id,
          assetPacks,
          runUid,
        });

        const phase = isWarmup ? "warmup" : "measure";
        console.log(
          `[clawbench-chain] ${phase}=${runIndex + 1}/${totalRuns} task=${taskRow.task_id} workspace=${workspaceDir}`,
        );

        await stageCapabilityWorkspaceForAgents(workspaceDir, taskRow);

        const taskBodyRaw = await resolveTaskBody(taskRow, runIndex);
        const taskBody = resolveTaskBodyForBench(taskBodyRaw, taskRow, args.benchPadding);
        const effectiveInferenceMode = isWarmup ? "dense_prefill" : args.inferenceMode;
        const result = await runChainStackSpawn(client, {
          agentId: args.agentId,
          scenario,
          taskRow: { ...taskRow, task_body: taskBody },
          model: args.model || undefined,
          runTimeoutSeconds: args.runTimeoutSeconds,
          experimentId: args.experimentId,
          runId,
          spawnMode: "capability",
          workspaceDir,
          inferenceMode: effectiveInferenceMode,
          inferenceBackend: args.inferenceBackend ?? runMetadata.inference_backend,
          taskProfile: "clawbench",
        });

        if (!isWarmup) {
          measureIndex += 1;
        }

        if (isWarmup) {
          for (const record of result.records) {
            await appendJsonl(
              outputPath,
              enrichAgentRecord(
                { ...record, workspace_dir: workspaceDir },
                runMetadata,
                runUid,
              ),
            );
          }
          await appendJsonl(
            outputPath,
            enrichRunSummary(
              {
                type: "run_summary",
                task_id: taskRow.task_id,
                run_id: runId,
                e2e_run_ms: result.e2e_run_ms,
                workspace_dir: workspaceDir,
                capability_score: null,
                orchestrator_session_keys: result.orchestrator_session_keys,
                timestamp: new Date().toISOString(),
                ...computeRunKvReuseStats(result.records),
              },
              runMetadata,
              runUid,
            ),
          );
        } else {
          const sessionMessages = await collectSessionMessages(client, result.records);
          const transcript = buildChainTranscript(taskBody, result.records, sessionMessages);
          await syncCapabilityWorkspaceArtifacts(workspaceDir, result.records, taskRow);

          let capabilityScore = null;
          if (!args.skipScore) {
            capabilityScore = slimCapabilityScore(
              await scoreCapabilityRun({
                taskId: taskRow.task_id,
                workspaceDir,
                transcript,
                judgeModel: args.judgeModel,
              }),
            );
          }

          for (const record of result.records) {
            await appendJsonl(
              outputPath,
              enrichAgentRecord(
                { ...record, workspace_dir: workspaceDir },
                runMetadata,
                runUid,
              ),
            );
          }
          await appendJsonl(
            outputPath,
            enrichRunSummary(
              {
                type: "run_summary",
                task_id: taskRow.task_id,
                run_id: runId,
                e2e_run_ms: result.e2e_run_ms,
                workspace_dir: workspaceDir,
                capability_score: capabilityScore,
                orchestrator_session_keys: result.orchestrator_session_keys,
                timestamp: new Date().toISOString(),
                ...computeRunKvReuseStats(result.records),
              },
              runMetadata,
              runUid,
            ),
          );
        }
      }
    }
  } finally {
    await client.close();
    await teardownManagedSidecar(sidecarHandle);
  }

  if (measureRuns === 0) {
    console.log("[clawbench-chain] warmup-only; no summary written");
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
  console.log("[clawbench-chain] results:", outputPath);
  console.log("[clawbench-chain] summary:", summaryPath);
  console.log("[clawbench-chain] report:", reportPath);
  if (summary.clawbench_capability) {
    const cap = summary.clawbench_capability;
    if (cap.by_measure?.length) {
      console.log("[clawbench-chain] clawbench capability by measure:");
      for (const row of cap.by_measure) {
        const status = row.passed ? "PASS" : (row.run_score ?? 0) >= 0.4 ? "PARTIAL" : "FAIL";
        console.log(
          `  [${row.measure_index}] run_uid=${row.run_uid} ` +
            `run=${row.run_score} C=${row.completion_score} ` +
            `T=${row.trajectory_score} B=${row.behavior_score} (${status})`,
        );
      }
      console.log(
        `[clawbench-chain] clawbench capability aggregate (${cap.runs} measure(s)): ` +
          `run=${cap.run_score_avg} C=${cap.completion_score_avg} ` +
          `T=${cap.trajectory_score_avg} B=${cap.behavior_score_avg} ` +
          `pass_rate=${Math.round((cap.pass_rate ?? 0) * 100)}%`,
      );
    } else if (cap.scoring_errors?.length) {
      console.log(`[clawbench-chain] capability scoring errors: ${cap.scoring_errors.length}`);
    }
  }
}

main().catch((err) => {
  console.error("[clawbench-chain] fatal:", err);
  process.exit(1);
});
