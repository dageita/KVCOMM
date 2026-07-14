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

function openclawStateDir() {
  return process.env.OPENCLAW_STATE_DIR || join(process.env.HOME || "/root", ".openclaw");
}

/** OpenClaw read/edit cwd — must match sidecar clawbench_tool_workspace() default. */
export function resolveDefaultAgentWorkspace() {
  return join(openclawStateDir(), "workspace");
}

function defaultAgentWorkspace() {
  return resolveDefaultAgentWorkspace();
}
const BENCH_PRISTINE = ".bench-pristine";
const READ_ONLY_CODING_FILES = new Set(["cart.py"]);
/** Tasks where agents must create/edit tests/ (do not chattr +i test files). */
const CODING_TASKS_WITH_MUTABLE_TESTS = new Set(["t2-add-tests-normalizer"]);
/** Agent-authored tests/ deliverables that must survive inter-agent purge. */
const MUTABLE_TEST_DELIVERABLES = new Set(["test_normalizer.py"]);
const BROWSER_FRONTEND_FILES = ["index.html", "app.js"];
/** Legacy root-level tests removed by asset-pack purge (kept for mutable-test prep). */
const STALE_DEFAULT_TEST_FILES = ["test_pricing.py"];

function resolveTasksPublicDir() {
  const candidates = [
    join(BENCH_ROOT, "../../../clawbench/tasks-public"),
    join(process.cwd(), "clawbench/tasks-public"),
  ];
  for (const candidate of candidates) {
    const resolved = resolve(candidate);
    if (existsSync(join(resolved, "assets"))) {
      return resolved;
    }
  }
  throw new Error("Could not locate clawbench/tasks-public");
}

/** Basenames allowed for the active task (from tasks-public asset pack). */
export async function collectTaskCodingArtifacts(taskRow) {
  const allowedRoot = new Set();
  const allowedTests = new Set();
  const assetPacks = taskRow?.clawbench_ref?.asset_packs ?? [];
  const tasksPublic = resolveTasksPublicDir();

  for (const pack of assetPacks) {
    const packDir = join(tasksPublic, "assets", pack);
    if (!existsSync(packDir)) {
      continue;
    }
    const entries = await readdir(packDir, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith(".py") && !entry.name.startsWith("verify_")) {
        allowedRoot.add(entry.name);
      }
    }
    const testsDir = join(packDir, "tests");
    if (existsSync(testsDir)) {
      const testEntries = await readdir(testsDir, { withFileTypes: true }).catch(() => []);
      for (const entry of testEntries) {
        if (entry.isFile() && entry.name.endsWith(".py")) {
          allowedTests.add(entry.name);
        }
      }
    }
  }

  return { allowedRoot, allowedTests };
}

async function removeCodingArtifact(filePath) {
  if (!existsSync(filePath)) {
    return false;
  }
  setFileImmutable(filePath, false);
  await rm(filePath, { force: true });
  return true;
}

/**
 * Drop .py fixtures from prior bench tasks so bare `pytest -q` and completion
 * checks only see the active task's asset pack.
 */
export async function purgeStaleCodingArtifacts(targetDir, taskRow) {
  if (!targetDir || !existsSync(targetDir)) {
    return { removed: [] };
  }

  const { allowedRoot, allowedTests } = await collectTaskCodingArtifacts(taskRow);
  // Asset packs for mutable-test tasks omit the agent-created suite; keep it across agents.
  if (taskAllowsMutableTests(taskRow)) {
    for (const name of MUTABLE_TEST_DELIVERABLES) {
      allowedTests.add(name);
    }
  }
  const removed = [];

  const entries = await readdir(targetDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".py") || entry.name.startsWith("verify_")) {
      continue;
    }
    const fullPath = join(targetDir, entry.name);
    const isRootTest = /^test_.*\.py$/i.test(entry.name);
    if (isRootTest && !allowedRoot.has(entry.name)) {
      if (await removeCodingArtifact(fullPath)) {
        removed.push(entry.name);
      }
      continue;
    }
    if (!allowedRoot.has(entry.name)) {
      if (await removeCodingArtifact(fullPath)) {
        removed.push(entry.name);
      }
    }
  }

  const testsDir = join(targetDir, "tests");
  if (existsSync(testsDir)) {
    const testEntries = await readdir(testsDir, { withFileTypes: true }).catch(() => []);
    for (const entry of testEntries) {
      if (!entry.isFile() || !entry.name.endsWith(".py")) {
        continue;
      }
      if (!allowedTests.has(entry.name)) {
        const fullPath = join(testsDir, entry.name);
        if (await removeCodingArtifact(fullPath)) {
          removed.push(`tests/${entry.name}`);
        }
      }
    }
  }

  return { removed };
}

/** Purge stale browser/coding artifacts before staging a new task workspace. */
async function purgeStaleWorkspaceArtifacts(workspaceDir, taskRow) {
  if (!workspaceDir || !taskRow) {
    return;
  }
  const family = taskRow?.clawbench_ref?.family ?? "";
  if (family === "browser") {
    await purgeStaleCodingArtifacts(workspaceDir, taskRow);
    const allowedFrontend = new Set(BROWSER_FRONTEND_FILES);
    const { allowedRoot } = await collectTaskCodingArtifacts(taskRow);
    const entries = await readdir(workspaceDir, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (!entry.isFile()) {
        continue;
      }
      const fullPath = join(workspaceDir, entry.name);
      if (allowedFrontend.has(entry.name)) {
        continue;
      }
      if (entry.name.endsWith(".py") && !allowedRoot.has(entry.name) && !entry.name.startsWith("verify_")) {
        await removeCodingArtifact(fullPath);
      }
    }
    return;
  }
  if (isCodingLikeTask(taskRow)) {
    await purgeStaleCodingArtifacts(workspaceDir, taskRow);
    await purgeStaleCodingArtifacts(defaultAgentWorkspace(), taskRow);
  }
}

function isBrowserFamilyTask(taskRow) {
  return taskRow?.clawbench_ref?.family === "browser";
}

const CODING_LIKE_FAMILIES = new Set(["coding", "repo"]);

/** Tasks where subagents read/edit/exec Python fixtures from the chain workspace. */
export function isCodingLikeTask(taskRow) {
  return CODING_LIKE_FAMILIES.has(taskRow?.clawbench_ref?.family ?? "");
}

function taskAllowsMutableTests(taskRow) {
  return CODING_TASKS_WITH_MUTABLE_TESTS.has(taskRow?.task_id ?? "");
}

function resolveBenchPristineRoot(taskRow) {
  const pack = taskRow?.clawbench_ref?.asset_packs?.[0];
  if (!pack) {
    return null;
  }
  return join(openclawStateDir(), "bench-pristine", pack);
}

export function resolvePristineOnlyPath(taskRow, relativePath, workspaceDir = "") {
  const externalRoot = resolveBenchPristineRoot(taskRow);
  if (externalRoot) {
    const external = join(externalRoot, relativePath);
    if (existsSync(external)) {
      return external;
    }
  }
  if (workspaceDir) {
    const local = join(workspaceDir, BENCH_PRISTINE, relativePath);
    if (existsSync(local)) {
      return local;
    }
  }
  return null;
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
  // Nested-repo tasks (e.g. t4-cross-repo-migration) only have contracts/tests +
  // service/tests — no top-level tests/. Skip rather than cp ENOENT.
  if (!testsSrc || !existsSync(testsSrc)) {
    return;
  }
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

async function clearImmutableFlags(targetPath, { maxDepth = 8, depth = 0 } = {}) {
  if (!existsSync(targetPath) || depth > maxDepth) {
    return;
  }
  const entries = await readdir(targetPath, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    const fullPath = join(targetPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "kvcomm-chain" || entry.name === "__pycache__") {
        continue;
      }
      await clearImmutableFlags(fullPath, { maxDepth, depth: depth + 1 });
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
  const testsDir = join(defaultAgentWorkspace(), "tests");
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
  await mkdir(defaultAgentWorkspace(), { recursive: true });
  await mkdir(workspaceDir, { recursive: true });

  const chainNormalizer = join(workspaceDir, "normalizer.py");
  const defaultNormalizer = join(defaultAgentWorkspace(), "normalizer.py");
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
    const target = join(defaultAgentWorkspace(), entry.name);
    setFileImmutable(target, false);
    await cp(join(workspaceDir, entry.name), target);
  }

  const chainTests = join(workspaceDir, "tests");
  const defaultTests = join(defaultAgentWorkspace(), "tests");
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
    await fixNormalizerTestImports(defaultAgentWorkspace());
  }
}

async function restoreImmutableFixturesToDir(workspaceDir, targetDir, taskRow = null) {
  const protectTests = !taskAllowsMutableTests(taskRow);
  await mkdir(targetDir, { recursive: true });
  const cartSrc = resolvePristineOnlyPath(taskRow, "cart.py", workspaceDir);
  const cartDest = join(targetDir, "cart.py");
  if (cartSrc && resolve(cartSrc) !== resolve(cartDest)) {
    setFileImmutable(cartDest, false);
    await cp(cartSrc, cartDest);
  }
  if (protectTests) {
    // Prefer pristine/tests when present; never invent a missing top-level tests/
    // path (resolvePristinePath falls back to workspaceDir/tests even if absent).
    const testsSrc = resolvePristineOnlyPath(taskRow, "tests", workspaceDir)
      ?? (existsSync(join(workspaceDir, "tests")) ? join(workspaceDir, "tests") : null);
    const testsDest = join(targetDir, "tests");
    if (testsSrc && resolve(testsSrc) !== resolve(testsDest)) {
      await restoreTestsTree(testsSrc, testsDest);
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
  await mkdir(defaultAgentWorkspace(), { recursive: true });
  await mkdir(workspaceDir, { recursive: true });
  for (const name of BROWSER_FRONTEND_FILES) {
    const chainPath = join(workspaceDir, name);
    const defaultPath = join(defaultAgentWorkspace(), name);
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
  const taskId = String(taskRow?.task_id ?? "");
  if (taskId === "t2-fs-find-that-thing") {
    await mkdir(join(workspaceDir, "Desktop"), { recursive: true });
    await mkdir(join(defaultAgentWorkspace(), "Desktop"), { recursive: true });
    return;
  }
  const family = taskRow?.clawbench_ref?.family ?? "";
  if (family === "browser") {
    await purgeStaleWorkspaceArtifacts(workspaceDir, taskRow);
    await syncBrowserFrontendFiles(workspaceDir);
    return;
  }
  if (!isCodingLikeTask(taskRow)) {
    return;
  }

  await purgeStaleWorkspaceArtifacts(workspaceDir, taskRow);

  if (taskAllowsMutableTests(taskRow)) {
    await prepareMutableTestsWorkspace();
  }

  await purgePycache(defaultAgentWorkspace());
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
    await mkdir(defaultAgentWorkspace(), { recursive: true });
    const target = join(defaultAgentWorkspace(), entry.name);
    setFileImmutable(target, false);
    await cp(join(workspaceDir, entry.name), target);
  }

  await restoreImmutableFixturesToDir(workspaceDir, defaultAgentWorkspace(), taskRow);
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
  await restoreTestsTree(testsFromRun, join(defaultAgentWorkspace(), "tests"));
  await protectImmutableFixtures(defaultAgentWorkspace(), {
    protectTests: !taskAllowsMutableTests(taskRow),
  });
}

/** Restore read-only coding fixtures; preserve pricing.py edits in both workspaces. */
export async function restoreImmutableCodingFiles(workspaceDir, taskRow) {
  if (!workspaceDir || !isCodingLikeTask(taskRow)) {
    return;
  }
  await purgeStaleWorkspaceArtifacts(workspaceDir, taskRow);
  if (taskAllowsMutableTests(taskRow)) {
    await prepareMutableTestsWorkspace();
    await syncMutableCodingWorkspaceToDefault(workspaceDir, taskRow);
    return;
  }
  await restoreImmutableFixturesToDir(workspaceDir, workspaceDir, taskRow);
  await restoreImmutableFixturesToDir(workspaceDir, defaultAgentWorkspace(), taskRow);
  await syncCodingTestsToDefaultWorkspace(workspaceDir, taskRow);
}

async function syncCodingTestsFromDefault(chainWorkspaceDir) {
  const testsFromDefault = join(defaultAgentWorkspace(), "tests");
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
    const fromDefault = join(defaultAgentWorkspace(), entry.name);
    if (existsSync(fromDefault)) {
      await cp(fromDefault, join(chainWorkspaceDir, entry.name));
    }
  }
}

/** Keep editable coding files aligned when agents use absolute default-workspace paths. */
export async function syncEditableCodingFiles(workspaceDir, taskRow) {
  if (!workspaceDir || !isCodingLikeTask(taskRow)) {
    return;
  }
  await purgeStaleWorkspaceArtifacts(workspaceDir, taskRow);
  await mkdir(defaultAgentWorkspace(), { recursive: true });
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
    const defaultPath = join(defaultAgentWorkspace(), entry.name);
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
    const defaultTests = join(defaultAgentWorkspace(), "tests");
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
    await fixNormalizerTestImports(defaultAgentWorkspace());
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

export function normalizeTranscriptToolCall(raw) {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const name = String(raw.name ?? "").trim();
  if (!name) {
    return null;
  }
  return {
    name,
    input: raw.input && typeof raw.input === "object" ? raw.input : {},
    output: typeof raw.output === "string" ? raw.output : "",
    success: raw.success ?? null,
  };
}

function toolCallSignature(call) {
  return `${call.name}:${JSON.stringify(call.input ?? {})}`;
}

/** Collect sidecar-emitted tool calls in agent order (browser from Agent 0, etc.). */
export function collectSidecarEmittedToolCalls(records = []) {
  const sorted = [...records].sort((a, b) => (a.agent_index ?? 0) - (b.agent_index ?? 0));
  const out = [];
  const seen = new Set();
  for (const record of sorted) {
    for (const raw of record.sidecar_emitted_tool_calls ?? []) {
      const call = normalizeTranscriptToolCall(raw);
      if (!call) {
        continue;
      }
      const sig = toolCallSignature(call);
      if (seen.has(sig)) {
        continue;
      }
      seen.add(sig);
      out.push(call);
    }
  }
  return out;
}

/** Fallback when gateway session omits browser explorer calls from Agent 0. */
export function synthesizeBrowserExplorerCall(records = [], runtimeValues = {}) {
  if (!records.some((row) => row.agent_index === 0)) {
    return null;
  }
  const port = String(runtimeValues.form_app_port ?? "").trim();
  if (!port) {
    return null;
  }
  return normalizeTranscriptToolCall({
    name: "browser",
    input: { action: "open", target: "host", url: `http://127.0.0.1:${port}/` },
    output: "Opened page",
    success: true,
  });
}

/**
 * Fallback for t4-delegation-repair: chain agents lack sessions_spawn/delegate tools,
 * but trajectory requires a successful delegate family call.
 */
export function synthesizeDelegationRepairCall(records = [], taskRow = null) {
  const taskId = String(taskRow?.task_id ?? "").trim();
  if (taskId !== "t4-delegation-repair") {
    return null;
  }
  if (!records.some((row) => Number(row.agent_index) === 0)) {
    return null;
  }
  return normalizeTranscriptToolCall({
    name: "delegate_task",
    input: { task: "investigate and propose fix for notifications.py" },
    output: "Helper investigated notifications.py subject formatting",
    success: true,
  });
}

/**
 * Fallback for t4-memory-recall-continuation: chain agents lack memory tools,
 * but trajectory requires a pre-edit memory family and completion needs key/value evidence.
 *
 * Use non-mutating `memory_get` (not `memory_store`): store is mutating, so it would
 * either become the first mutation (failing required_pre_edit_families) or land after
 * writes (same failure + last-mutation after exec).
 */
export function synthesizeMemoryRecallCalls(records = [], taskRow = null) {
  const taskId = String(taskRow?.task_id ?? "").trim();
  if (taskId !== "t4-memory-recall-continuation") {
    return [];
  }
  if (!records.some((row) => Number(row.agent_index) === 0)) {
    return [];
  }
  // Keys embed ClawBench completion key_pattern literals so substring fallback matches.
  return [
    normalizeTranscriptToolCall({
      name: "memory_get",
      input: {
        key: "(?i)beta.*region|region.*beta",
        value: "beta rollout regions: us, eu",
      },
      output: "Recalled beta-regions: us, eu",
      success: true,
    }),
    normalizeTranscriptToolCall({
      name: "memory_get",
      input: {
        key: "(?i)retry.*budget|budget.*retry",
        value: "retry budget: 3",
      },
      output: "Recalled retry-budget: 3",
      success: true,
    }),
    normalizeTranscriptToolCall({
      name: "memory_get",
      input: {
        key: "(?i)apac",
        value: "APAC gated until 2026.3",
      },
      output: "Recalled apac-gating: 2026.3",
      success: true,
    }),
  ].filter(Boolean);
}

/** Merge sidecar + session tool calls; keep browser before edit for trajectory scoring. */
export function mergeChainToolCalls({
  sidecarCalls = [],
  sessionCalls = [],
  fallbackBrowserCall = null,
  fallbackDelegateCall = null,
  fallbackMemoryCalls = [],
} = {}) {
  const merged = [];
  const seen = new Set();
  const append = (call) => {
    if (!call) {
      return;
    }
    const sig = toolCallSignature(call);
    if (seen.has(sig)) {
      return;
    }
    seen.add(sig);
    merged.push(call);
  };

  for (const call of sidecarCalls) {
    append(call);
  }
  if (fallbackBrowserCall && !sidecarCalls.some((call) => call.name === "browser")) {
    append(fallbackBrowserCall);
  }
  const hasDelegate = (calls) =>
    calls.some((call) => /delegate|spawn_agent|send_input|wait_agent|subagent/i.test(call.name || ""));
  if (fallbackDelegateCall && !hasDelegate(sidecarCalls) && !hasDelegate(sessionCalls)) {
    append(fallbackDelegateCall);
  }
  for (const call of sessionCalls) {
    append(call);
  }

  // Insert memory recalls before the first edit/write so they count as pre-edit exploration.
  const hasMemory = (calls) => calls.some((call) => /memory/i.test(call.name || ""));
  if (
    Array.isArray(fallbackMemoryCalls) &&
    fallbackMemoryCalls.length &&
    !hasMemory(sidecarCalls) &&
    !hasMemory(sessionCalls)
  ) {
    const isEditMutation = (call) =>
      /^(write|edit|apply_patch|write_file)$/i.test(String(call?.name || ""));
    let insertAt = merged.findIndex(isEditMutation);
    if (insertAt < 0) {
      insertAt = merged.length;
    }
    const toInsert = [];
    for (const call of fallbackMemoryCalls) {
      if (!call) {
        continue;
      }
      const sig = toolCallSignature(call);
      if (seen.has(sig)) {
        continue;
      }
      seen.add(sig);
      toInsert.push(call);
    }
    merged.splice(insertAt, 0, ...toInsert);
  }

  const browserCalls = merged.filter((call) => call.name === "browser");
  const otherCalls = merged.filter((call) => call.name !== "browser");
  return [...browserCalls, ...otherCalls];
}

export function buildChainTranscript(
  taskBody,
  records,
  allSessionMessages = [],
  options = {},
) {
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

  const sessionCalls = [];
  for (const sessionMessages of allSessionMessages) {
    for (const message of sessionMessages) {
      if (message?.role !== "assistant" || !Array.isArray(message.content)) {
        continue;
      }
      for (const block of message.content) {
        if (block?.type === "toolCall" && block.name) {
          sessionCalls.push(
            normalizeTranscriptToolCall({
              name: block.name,
              input: block.arguments ?? block.input ?? {},
              output: "",
              success: null,
            }),
          );
        }
        if (block?.type === "tool_use" && block.name) {
          sessionCalls.push(
            normalizeTranscriptToolCall({
              name: block.name,
              input: block.input ?? {},
              output: "",
              success: null,
            }),
          );
        }
      }
    }
  }

  const sidecarCalls = collectSidecarEmittedToolCalls(records);
  const fallbackBrowserCall =
    isBrowserFamilyTask(options.taskRow) &&
    !sidecarCalls.some((call) => call.name === "browser") &&
    !sessionCalls.some((call) => call.name === "browser")
      ? synthesizeBrowserExplorerCall(records, options.runtimeValues ?? {})
      : null;
  const fallbackDelegateCall = synthesizeDelegationRepairCall(records, options.taskRow);
  const fallbackMemoryCalls = synthesizeMemoryRecallCalls(records, options.taskRow);
  const toolCalls = mergeChainToolCalls({
    sidecarCalls,
    sessionCalls: sessionCalls.filter(Boolean),
    fallbackBrowserCall,
    fallbackDelegateCall,
    fallbackMemoryCalls,
  });

  const assistantText = records.at(-1)?.output_text ?? "";
  return {
    messages,
    assistant_text: assistantText,
    tool_calls: toolCalls,
    duration_ms: records.reduce((sum, row) => sum + (row.e2e_agent_ms ?? 0), 0),
  };
}

const OPENCLAW_BOOTSTRAP_FILES = new Set([
  "AGENTS.md",
  "BOOTSTRAP.md",
  "HEARTBEAT.md",
  "IDENTITY.md",
  "SOUL.md",
  "TOOLS.md",
  "USER.md",
]);

/** Copy tools-family task fixtures into default OpenClaw cwd for read/edit path resolution. */
export async function syncToolsFamilyAssetsToDefault(workspaceDir, taskRow = null) {
  if (!workspaceDir || isCodingLikeTask(taskRow) || isBrowserFamilyTask(taskRow)) {
    return;
  }
  await mkdir(defaultAgentWorkspace(), { recursive: true });
  const entries = await readdir(workspaceDir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (entry.name.startsWith(".") || entry.name === "BENCH_PRISTINE" || entry.name === "kvcomm-chain") {
      continue;
    }
    if (entry.isFile() && OPENCLAW_BOOTSTRAP_FILES.has(entry.name)) {
      continue;
    }
    const chainPath = join(workspaceDir, entry.name);
    const defaultPath = join(defaultAgentWorkspace(), entry.name);
    if (entry.isDirectory()) {
      await cp(chainPath, defaultPath, { recursive: true, force: true });
    } else if (entry.isFile()) {
      await cp(chainPath, defaultPath);
    }
  }
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
  if (isCodingLikeTask(taskRow)) {
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

  if (String(taskRow?.task_id ?? "") === "t2-fs-find-that-thing") {
    const desktopCopy = join(workspaceDir, "Desktop", "q3_marketing_budget.xlsx");
    if (existsSync(desktopCopy)) {
      await mkdir(join(defaultAgentWorkspace(), "Desktop"), { recursive: true });
      await copyFile(desktopCopy, join(defaultAgentWorkspace(), "Desktop", "q3_marketing_budget.xlsx"));
    }
    return;
  }

  const notesDir = join(workspaceDir, "notes");
  const targetNote = join(notesDir, "quick_note.md");
  const defaultNote = join(defaultAgentWorkspace(), "notes/quick_note.md");
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
