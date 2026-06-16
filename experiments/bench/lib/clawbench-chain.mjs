import { mkdir, mkdtemp, readFile, rm, writeFile, copyFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { extractAssistantText } from "./gateway-client.mjs";
import { resolveGatewayToken } from "./openclaw-config.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = join(__dirname, "..");

export function setupClawbenchWorkspace({ taskId, assetPacks, runUid }) {
  const script = join(BENCH_ROOT, "scripts/setup-clawbench-workspace.py");
  const result = spawnSync(
    "python3",
    [
      script,
      "--task-id",
      taskId,
      "--asset-packs",
      assetPacks.join(","),
      "--run-id",
      runUid,
      "--output-json",
    ],
    { encoding: "utf8" },
  );
  if (result.status !== 0) {
    throw new Error(
      `setup-clawbench-workspace failed: ${result.stderr || result.stdout || result.status}`,
    );
  }
  const payload = JSON.parse(result.stdout.trim());
  return payload.workspace;
}

export function buildChainTranscript(taskBody, records, allSessionMessages = []) {
  const messages = [{ role: "user", text: taskBody }];
  for (const record of records) {
    if (record.output_text) {
      messages.push({
        role: "assistant",
        text: record.output_text,
        agent_index: record.agent_index,
      });
    }
  }

  const toolCalls = [];
  for (const sessionMessages of allSessionMessages) {
    for (const message of sessionMessages) {
      if (message?.role !== "assistant" || !Array.isArray(message.content)) {
        continue;
      }
      for (const block of message.content) {
        if (block?.type === "toolCall" && block.name) {
          toolCalls.push({
            name: block.name,
            input: block.arguments ?? block.input ?? {},
            output: "",
            success: null,
          });
        }
        if (block?.type === "tool_use" && block.name) {
          toolCalls.push({
            name: block.name,
            input: block.input ?? {},
            output: "",
            success: null,
          });
        }
      }
    }
  }

  const assistantText = records.at(-1)?.output_text ?? "";
  return {
    messages,
    assistant_text: assistantText,
    tool_calls: toolCalls,
    duration_ms: records.reduce((sum, row) => sum + (row.e2e_agent_ms ?? 0), 0),
  };
}

export async function syncCapabilityWorkspaceArtifacts(workspaceDir, records) {
  if (!workspaceDir) {
    return;
  }
  const notesDir = join(workspaceDir, "notes");
  const targetNote = join(notesDir, "quick_note.md");
  const defaultNote = join(process.env.HOME || "/root", ".openclaw/workspace/notes/quick_note.md");
  if (existsSync(defaultNote)) {
    await mkdir(notesDir, { recursive: true });
    await copyFile(defaultNote, targetNote);
    return;
  }
  const writer = records.find((row) => row.agent_index === 1);
  const text = writer?.output_text ?? "";
  const listLines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^[-*+]\s/.test(line) || /^\d+[.)]\s/.test(line) || /^- \[[ x]\]\s/.test(line));
  if (listLines.length >= 3) {
    await mkdir(notesDir, { recursive: true });
    await writeFile(targetNote, `${listLines.join("\n")}\n`, "utf8");
  }
}

function roundScore(value) {
  if (value == null || Number.isNaN(value)) {
    return null;
  }
  return Math.round(value * 1000) / 1000;
}

/** Strip scorer payload to fields stored in bench jsonl / summary.json. */
export function slimCapabilityScore(score) {
  if (!score) {
    return null;
  }
  if (score.error) {
    return { error: score.error };
  }
  return {
    task_id: score.task_id,
    run_score: roundScore(score.run_score),
    completion_score: roundScore(score.completion_score),
    trajectory_score: roundScore(score.trajectory_score),
    behavior_score: roundScore(score.behavior_score),
    judge_score: roundScore(score.judge_score),
    judge_error: score.judge_error ?? null,
    failed_assertions: score.failed_assertions ?? [],
    trajectory_violations: score.trajectory_violations ?? [],
  };
}

/** Format ClawBench native scores for console (matches clawbench harness C/T/B labels). */
export function formatCapabilityScoreReport(
  score,
  { taskId = "", runUid = "", measureIndex = null } = {},
) {
  const measureLabel = measureIndex == null ? "" : ` measure=${measureIndex}`;
  const prefix = `[clawbench-score] ${taskId}${measureLabel}${runUid ? ` run=${runUid}` : ""}`;
  if (!score || score.error) {
    return [`${prefix}: scoring failed — ${score?.error ?? "unknown error"}`];
  }

  const run = roundScore(score.run_score);
  const completion = roundScore(score.completion_score);
  const trajectory = roundScore(score.trajectory_score);
  const behavior = roundScore(score.behavior_score);
  const judge = roundScore(score.judge_score);
  const passed = (run ?? 0) >= 0.7;
  const status = passed ? "PASS" : (run ?? 0) >= 0.4 ? "PARTIAL" : "FAIL";

  const lines = [
    `${prefix}: run=${run} C=${completion} T=${trajectory} B=${behavior}` +
      (judge != null ? ` J=${judge}` : "") +
      ` (${status})`,
  ];
  if (score.judge_error) {
    lines.push(`${prefix}: judge unavailable — ${score.judge_error}`);
  }
  for (const failure of score.failed_assertions ?? []) {
    lines.push(`${prefix}: completion ! ${failure}`);
  }
  for (const violation of score.trajectory_violations ?? []) {
    lines.push(`${prefix}: trajectory ! ${violation}`);
  }
  return lines;
}

export async function scoreCapabilityRun({
  taskId,
  workspaceDir,
  transcript,
  judgeModel = "",
}) {
  const tempDir = await mkdtemp(join(tmpdir(), "kvcomm-clawbench-score-"));
  const transcriptPath = join(tempDir, "transcript.json");
  const capabilityPath = join(tempDir, "capability.json");
  await writeFile(transcriptPath, `${JSON.stringify(transcript, null, 2)}\n`, "utf8");

  const script = join(BENCH_ROOT, "scripts/score-clawbench-chain-run.py");
  const clawbenchRoot = resolve(BENCH_ROOT, "../../../clawbench");
  const args = [
    script,
    "--task-id",
    taskId,
    "--workspace",
    workspaceDir,
    "--transcript",
    transcriptPath,
    "--output",
    capabilityPath,
  ];
  if (judgeModel) {
    args.push("--judge-model", judgeModel);
  }

  const { token } = await resolveGatewayToken();
  const env = { ...process.env, PYTHONPATH: clawbenchRoot };
  if (token) {
    env.OPENCLAW_GATEWAY_TOKEN = token;
  }

  const result = spawnSync("uv", ["run", "python", ...args], {
    encoding: "utf8",
    cwd: clawbenchRoot,
    env,
  });
  try {
    if (result.status !== 0) {
      return {
        error: result.stderr || result.stdout || `exit ${result.status}`,
      };
    }
    const payload = JSON.parse(await readFile(capabilityPath, "utf8"));
    return payload;
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

export async function collectSessionMessages(client, records) {
  const all = [];
  for (const record of records) {
    if (!record.child_session_key) {
      continue;
    }
    const messages = await client.getSessionMessages(record.child_session_key);
    all.push(messages);
    if (!record.output_text) {
      record.output_text = extractAssistantText(messages);
    }
  }
  return all;
}
