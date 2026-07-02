import { cp, mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile, copyFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { extractAssistantText } from "./gateway-client.mjs";
import { resolveGatewayToken } from "./openclaw-config.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = join(__dirname, "..");
const DEFAULT_AGENT_WORKSPACE = join(process.env.HOME || "/root", ".openclaw/workspace");
const BENCH_PRISTINE = ".bench-pristine";
const READ_ONLY_CODING_FILES = new Set(["cart.py"]);
/** Tasks where agents must create/edit tests/ (do not chattr +i test files). */
const CODING_TASKS_WITH_MUTABLE_TESTS = new Set(["t2-add-tests-normalizer"]);
const BROWSER_FRONTEND_FILES = ["index.html", "app.js"];
const STALE_DEFAULT_TEST_FILES = ["test_pricing.py"];

function isBrowserFamilyTask(taskRow) {
  return taskRow?.clawbench_ref?.family === "browser";
}

function taskAllowsMutableTests(taskRow) {
  return CODING_TASKS_WITH_MUTABLE_TESTS.has(taskRow?.task_id ?? "");
}

function openclawStateDir() {
  return process.env.OPENCLAW_STATE_DIR || join(process.env.HOME || "/root", ".openclaw");
}

function resolveBenchPristineRoot(taskRow) {
  const pack = taskRow?.clawbench_ref?.asset_packs?.[0];
  if (!pack) {
    return null;
  }
  return join(openclawStateDir(), "bench-pristine", pack);
}

function resolvePristinePath(workspaceDir, relativePath, taskRow = null) {
  const externalRoot = resolveBenchPristineRoot(taskRow);
  if (externalRoot) {
    const external = join(externalRoot, relativePath);
    if (existsSync(external)) {
      return external;
    }
  }
  const local = join(workspaceDir, BENCH_PRISTINE, relativePath);
  if (existsSync(local)) {
    return local;
  }
  return join(workspaceDir, relativePath);
}

async function restoreTestsTree(testsSrc, targetTestsDir) {
  await clearImmutableFlags(targetTestsDir);
  await rm(targetTestsDir, { recursive: true, force: true });
  await cp(testsSrc, targetTestsDir, {
    recursive: true,
    filter: (src) => !src.split("/").includes("__pycache__") && !src.endsWith(".pyc"),
  });
}

function setFileImmutable(filePath, immutable) {
  if (!existsSync(filePath)) {
    return;
  }
  spawnSync("chattr", [immutable ? "+i" : "-i", filePath], { stdio: "ignore" });
}

async function clearImmutableFlags(targetPath) {
  if (!existsSync(targetPath)) {
    return;
  }
  const entries = await readdir(targetPath, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    const fullPath = join(targetPath, entry.name);
    if (entry.isDirectory()) {
      await clearImmutableFlags(fullPath);
    } else {
      setFileImmutable(fullPath, false);
    }
  }
}

async function protectImmutableFixtures(targetDir, { protectTests = true } = {}) {
  const cartPath = join(targetDir, "cart.py");
  if (existsSync(cartPath)) {
    setFileImmutable(cartPath, true);
  }
  if (!protectTests) {
    return;
  }
  const testsDir = join(targetDir, "tests");
  if (!existsSync(testsDir)) {
    return;
  }
  const entries = await readdir(testsDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith(".py")) {
      setFileImmutable(join(testsDir, entry.name), true);
    }
  }
}

async function prepareMutableTestsWorkspace() {
  const testsDir = join(DEFAULT_AGENT_WORKSPACE, "tests");
  await mkdir(testsDir, { recursive: true });
  await clearImmutableFlags(testsDir);
  for (const name of [...STALE_DEFAULT_TEST_FILES, "test_normalizer.py"]) {
    const stalePath = join(testsDir, name);
    if (existsSync(stalePath)) {
      setFileImmutable(stalePath, false);
      await rm(stalePath, { force: true });
    }
  }
}

/** Keep chain workspace and default OpenClaw workspace aligned for test-authoring tasks. */
async function fixNormalizerTestImports(targetDir) {
  const testPath = join(targetDir, "tests", "test_normalizer.py");
  if (!existsSync(testPath)) {
    return;
  }
  const content = await readFile(testPath, "utf8");
  const fixed = content
    .replace(/from\s+\.\.normalizer\s+import/gi, "from normalizer import")
    .replace(/from\s+\.\.\s+import\s+normalize_title,\s*normalize_tags/gi, "from normalizer import normalize_title, normalize_tags")
    .replace(/from\s+\.\.\s+import\s+normalizer\b/gi, "from normalizer import")
    .replace(/from\s+openclaw\.normalizer\s+import\s+normalize_text\b/gi, "from normalizer import normalize_title, normalize_tags")
    .replace(/from\s+openclaw\.normalizer\s+import/gi, "from normalizer import");
  if (fixed !== content) {
    await writeFile(testPath, fixed, "utf8");
  }
}

async function syncMutableCodingWorkspaceToDefault(workspaceDir, taskRow = null) {
  if (!workspaceDir) {
    return;
  }
  await mkdir(DEFAULT_AGENT_WORKSPACE, { recursive: true });
  await mkdir(workspaceDir, { recursive: true });

  const chainNormalizer = join(workspaceDir, "normalizer.py");
  const defaultNormalizer = join(DEFAULT_AGENT_WORKSPACE, "normalizer.py");
  if (!existsSync(chainNormalizer) && !existsSync(defaultNormalizer)) {
    const pristine = resolvePristinePath(workspaceDir, "normalizer.py", taskRow);
    if (existsSync(pristine)) {
      await cp(pristine, chainNormalizer);
      await cp(pristine, defaultNormalizer);
    }
  } else if (existsSync(chainNormalizer) && !existsSync(defaultNormalizer)) {
    await cp(chainNormalizer, defaultNormalizer);
  } else if (existsSync(defaultNormalizer) && !existsSync(chainNormalizer)) {
    await cp(defaultNormalizer, chainNormalizer);
  } else if (existsSync(chainNormalizer) && existsSync(defaultNormalizer)) {
    const [chainStat, defaultStat] = await Promise.all([stat(chainNormalizer), stat(defaultNormalizer)]);
    if (defaultStat.mtimeMs > chainStat.mtimeMs) {
      await cp(defaultNormalizer, chainNormalizer);
    } else if (chainStat.mtimeMs > defaultStat.mtimeMs) {
      await cp(chainNormalizer, defaultNormalizer);
    }
  }

  const entries = await readdir(workspaceDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (
      !entry.isFile()
      || !entry.name.endsWith(".py")
      || entry.name.startsWith("verify_")
      || READ_ONLY_CODING_FILES.has(entry.name)
    ) {
      continue;
    }
    const target = join(DEFAULT_AGENT_WORKSPACE, entry.name);
    setFileImmutable(target, false);
    await cp(join(workspaceDir, entry.name), target);
  }

  const chainTests = join(workspaceDir, "tests");
  const defaultTests = join(DEFAULT_AGENT_WORKSPACE, "tests");
  if (existsSync(chainTests)) {
    await clearImmutableFlags(defaultTests);
    await restoreTestsTree(chainTests, defaultTests);
  } else if (existsSync(defaultTests)) {
    await clearImmutableFlags(chainTests);
    await mkdir(workspaceDir, { recursive: true });
    await restoreTestsTree(defaultTests, chainTests);
  }

  if (taskAllowsMutableTests(taskRow)) {
    await fixNormalizerTestImports(workspaceDir);
    await fixNormalizerTestImports(DEFAULT_AGENT_WORKSPACE);
  }
}

async function restoreImmutableFixturesToDir(workspaceDir, targetDir, taskRow = null) {
  const protectTests = !taskAllowsMutableTests(taskRow);
  await mkdir(targetDir, { recursive: true });
  const cartSrc = resolvePristinePath(workspaceDir, "cart.py", taskRow);
  if (existsSync(cartSrc)) {
    setFileImmutable(join(targetDir, "cart.py"), false);
    await cp(cartSrc, join(targetDir, "cart.py"));
  }
  if (protectTests) {
    const testsSrc = resolvePristinePath(workspaceDir, "tests", taskRow);
    if (existsSync(testsSrc)) {
      await restoreTestsTree(testsSrc, join(targetDir, "tests"));
    }
  }
  await protectImmutableFixtures(targetDir, { protectTests });
}

async function purgePycache(targetPath) {
  if (!existsSync(targetPath)) {
    return;
  }
  const entries = await readdir(targetPath, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    const fullPath = join(targetPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "__pycache__") {
        await rm(fullPath, { recursive: true, force: true });
      } else {
        await purgePycache(fullPath);
      }
    } else if (entry.name.endsWith(".pyc")) {
      await rm(fullPath, { force: true });
    }
  }
}

/** Copy browser frontend files between chain workspace and OpenClaw default cwd. */
async function syncBrowserFrontendFiles(workspaceDir, { preferDefault = false } = {}) {
  if (!workspaceDir) {
    return;
  }
  await mkdir(DEFAULT_AGENT_WORKSPACE, { recursive: true });
  await mkdir(workspaceDir, { recursive: true });
  for (const name of BROWSER_FRONTEND_FILES) {
    const chainPath = join(workspaceDir, name);
    const defaultPath = join(DEFAULT_AGENT_WORKSPACE, name);
    const chainExists = existsSync(chainPath);
    const defaultExists = existsSync(defaultPath);
    if (chainExists && !defaultExists) {
      await cp(chainPath, defaultPath);
      continue;
    }
    if (!chainExists && defaultExists) {
      await cp(defaultPath, chainPath);
      continue;
    }
    if (!chainExists || !defaultExists) {
      continue;
    }
    const [chainStat, defaultStat] = await Promise.all([stat(chainPath), stat(defaultPath)]);
    if (preferDefault) {
      if (defaultStat.mtimeMs >= chainStat.mtimeMs) {
        await cp(defaultPath, chainPath);
      } else {
        await cp(chainPath, defaultPath);
      }
    } else if (defaultStat.mtimeMs > chainStat.mtimeMs) {
      await cp(defaultPath, chainPath);
    } else if (chainStat.mtimeMs > defaultStat.mtimeMs) {
      await cp(chainPath, defaultPath);
    }
  }
}

/** Keep editable browser files aligned when subagents read/edit in default workspace. */
export async function syncEditableBrowserFiles(workspaceDir, taskRow, { preferDefault = false } = {}) {
  if (!workspaceDir || !isBrowserFamilyTask(taskRow)) {
    return;
  }
  await syncBrowserFrontendFiles(workspaceDir, { preferDefault });
}

/** Copy chain workspace task files into OpenClaw default workspace (subagent read/edit cwd). */
export async function stageCapabilityWorkspaceForAgents(workspaceDir, taskRow) {
  if (!workspaceDir) {
    return;
  }
  const family = taskRow?.clawbench_ref?.family ?? "";
  if (family === "browser") {
    await syncBrowserFrontendFiles(workspaceDir);
    return;
  }
  if (family !== "coding") {
    return;
  }

  if (taskAllowsMutableTests(taskRow)) {
    await prepareMutableTestsWorkspace();
  }

  await purgePycache(DEFAULT_AGENT_WORKSPACE);
  const entries = await readdir(workspaceDir, { withFileTypes: true });
  for (const entry of entries) {
    if (
      !entry.isFile()
      || !entry.name.endsWith(".py")
      || entry.name.startsWith("verify_")
      || READ_ONLY_CODING_FILES.has(entry.name)
    ) {
      continue;
    }
    await mkdir(DEFAULT_AGENT_WORKSPACE, { recursive: true });
    const target = join(DEFAULT_AGENT_WORKSPACE, entry.name);
    setFileImmutable(target, false);
    await cp(join(workspaceDir, entry.name), target);
  }

  await restoreImmutableFixturesToDir(workspaceDir, DEFAULT_AGENT_WORKSPACE, taskRow);
  await syncCodingTestsToDefaultWorkspace(workspaceDir, taskRow);
  if (taskAllowsMutableTests(taskRow)) {
    await syncMutableCodingWorkspaceToDefault(workspaceDir, taskRow);
  }
}

async function syncCodingTestsToDefaultWorkspace(workspaceDir, taskRow = null) {
  const testsFromRun = join(workspaceDir, "tests");
  if (!existsSync(testsFromRun)) {
    return;
  }
  await restoreTestsTree(testsFromRun, join(DEFAULT_AGENT_WORKSPACE, "tests"));
  await protectImmutableFixtures(DEFAULT_AGENT_WORKSPACE, {
    protectTests: !taskAllowsMutableTests(taskRow),
  });

  const staleRootTest = join(DEFAULT_AGENT_WORKSPACE, "test_pricing.py");
  const canonicalTest = join(DEFAULT_AGENT_WORKSPACE, "tests", "test_pricing.py");
  if (existsSync(staleRootTest) && existsSync(canonicalTest)) {
    setFileImmutable(staleRootTest, false);
    await rm(staleRootTest, { force: true });
  }
}

/** Restore read-only coding fixtures; preserve pricing.py edits in both workspaces. */
export async function restoreImmutableCodingFiles(workspaceDir, taskRow) {
  if (!workspaceDir || taskRow?.clawbench_ref?.family !== "coding") {
    return;
  }
  if (taskAllowsMutableTests(taskRow)) {
    await prepareMutableTestsWorkspace();
    await syncMutableCodingWorkspaceToDefault(workspaceDir, taskRow);
    return;
  }
  await restoreImmutableFixturesToDir(workspaceDir, workspaceDir, taskRow);
  await restoreImmutableFixturesToDir(workspaceDir, DEFAULT_AGENT_WORKSPACE, taskRow);
  await syncCodingTestsToDefaultWorkspace(workspaceDir, taskRow);
}

async function syncCodingTestsFromDefault(chainWorkspaceDir) {
  const testsFromDefault = join(DEFAULT_AGENT_WORKSPACE, "tests");
  if (!existsSync(testsFromDefault)) {
    return;
  }
  await clearImmutableFlags(testsFromDefault);
  await restoreTestsTree(testsFromDefault, join(chainWorkspaceDir, "tests"));
}

async function syncCodingArtifactsFromDefault(chainWorkspaceDir) {
  const entries = await readdir(chainWorkspaceDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (
      !entry.isFile()
      || !entry.name.endsWith(".py")
      || entry.name.startsWith("verify_")
      || READ_ONLY_CODING_FILES.has(entry.name)
    ) {
      continue;
    }
    const fromDefault = join(DEFAULT_AGENT_WORKSPACE, entry.name);
    if (existsSync(fromDefault)) {
      await cp(fromDefault, join(chainWorkspaceDir, entry.name));
    }
  }
}

/** Keep editable coding files aligned when agents use absolute default-workspace paths. */
export async function syncEditableCodingFiles(workspaceDir, taskRow) {
  if (!workspaceDir || taskRow?.clawbench_ref?.family !== "coding") {
    return;
  }
  await mkdir(DEFAULT_AGENT_WORKSPACE, { recursive: true });
  const entries = await readdir(workspaceDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (
      !entry.isFile()
      || !entry.name.endsWith(".py")
      || entry.name.startsWith("verify_")
      || READ_ONLY_CODING_FILES.has(entry.name)
    ) {
      continue;
    }
    const chainPath = join(workspaceDir, entry.name);
    const defaultPath = join(DEFAULT_AGENT_WORKSPACE, entry.name);
    const chainExists = existsSync(chainPath);
    const defaultExists = existsSync(defaultPath);
    if (chainExists && !defaultExists) {
      await cp(chainPath, defaultPath);
      continue;
    }
    if (!chainExists && defaultExists) {
      await cp(defaultPath, chainPath);
      continue;
    }
    if (!chainExists || !defaultExists) {
      continue;
    }
    const [chainStat, defaultStat] = await Promise.all([stat(chainPath), stat(defaultPath)]);
    if (defaultStat.mtimeMs > chainStat.mtimeMs) {
      await cp(defaultPath, chainPath);
    } else if (chainStat.mtimeMs > defaultStat.mtimeMs) {
      await cp(chainPath, defaultPath);
    }
  }

  if (taskAllowsMutableTests(taskRow)) {
    const chainTests = join(workspaceDir, "tests");
    const defaultTests = join(DEFAULT_AGENT_WORKSPACE, "tests");
    if (existsSync(chainTests) && !existsSync(defaultTests)) {
      await restoreTestsTree(chainTests, defaultTests);
    } else if (existsSync(defaultTests) && !existsSync(chainTests)) {
      await restoreTestsTree(defaultTests, chainTests);
    } else if (existsSync(chainTests) && existsSync(defaultTests)) {
      const [chainStat, defaultStat] = await Promise.all([stat(chainTests), stat(defaultTests)]);
      if (defaultStat.mtimeMs > chainStat.mtimeMs) {
        await restoreTestsTree(defaultTests, chainTests);
      } else if (chainStat.mtimeMs > defaultStat.mtimeMs) {
        await restoreTestsTree(chainTests, defaultTests);
      }
    }
    await fixNormalizerTestImports(workspaceDir);
    await fixNormalizerTestImports(DEFAULT_AGENT_WORKSPACE);
  }
}

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

export async function syncCapabilityWorkspaceArtifacts(workspaceDir, records, taskRow = null) {
  if (!workspaceDir) {
    return;
  }

  const family = taskRow?.clawbench_ref?.family ?? "";
  if (family === "browser") {
    await syncEditableBrowserFiles(workspaceDir, taskRow);
    return;
  }
  if (family === "coding") {
    await syncEditableCodingFiles(workspaceDir, taskRow);
    if (!taskAllowsMutableTests(taskRow)) {
      await restoreImmutableCodingFiles(workspaceDir, taskRow);
    }
    await syncCodingArtifactsFromDefault(workspaceDir);
    if (taskAllowsMutableTests(taskRow)) {
      await syncCodingTestsFromDefault(workspaceDir);
    }
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

export function buildScoringRuntimeValues(workspaceDir, runtimeValues = {}) {
  const clawbenchRoot = resolve(BENCH_ROOT, "../../../clawbench");
  const openclawRoot = process.env.OPENCLAW_ROOT || join(process.env.HOME || "/root", ".openclaw");
  return {
    workspace: workspaceDir,
    workspace_name: workspaceDir.split("/").filter(Boolean).pop() ?? "",
    repo_root: clawbenchRoot,
    benchmark_node_path: join(clawbenchRoot, "node_modules"),
    openclaw_node_path: join(openclawRoot, "node_modules"),
    ...runtimeValues,
  };
}

export async function scoreCapabilityRun({
  taskId,
  workspaceDir,
  transcript,
  judgeModel = "",
  runtimeValues = {},
}) {
  const tempDir = await mkdtemp(join(tmpdir(), "kvcomm-clawbench-score-"));
  const transcriptPath = join(tempDir, "transcript.json");
  const runtimeValuesPath = join(tempDir, "runtime-values.json");
  const capabilityPath = join(tempDir, "capability.json");
  await writeFile(transcriptPath, `${JSON.stringify(transcript, null, 2)}\n`, "utf8");
  await writeFile(
    runtimeValuesPath,
    `${JSON.stringify(buildScoringRuntimeValues(workspaceDir, runtimeValues), null, 2)}\n`,
    "utf8",
  );

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
    "--runtime-values",
    runtimeValuesPath,
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
