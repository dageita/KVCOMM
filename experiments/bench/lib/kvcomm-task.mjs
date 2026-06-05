import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");
const DEFAULT_FIXTURE = join(BENCH_ROOT, "fixtures/kvcomm_tasks_seed42.json");

let fixtureCache = null;

async function loadFixture(relOrAbsPath) {
  const path = relOrAbsPath.startsWith("/") ? relOrAbsPath : join(BENCH_ROOT, relOrAbsPath);
  if (fixtureCache?.path === path) {
    return fixtureCache.data;
  }
  const raw = await readFile(path, "utf8");
  const data = JSON.parse(raw);
  if (!Array.isArray(data.tasks)) {
    throw new Error(`Invalid kvcomm task fixture (missing tasks[]): ${path}`);
  }
  fixtureCache = { path, data };
  return data;
}

/**
 * Resolve task body to match KVCOMM benchmark_TTFT.py (SEED=42, 1000 Δ/Ω symbols, space-separated).
 *
 * Dataset fields (first match wins):
 *   - task_body: fixed string
 *   - kvcomm_task: { fixture?, index?, advance_per_run? }
 *   - user_question: legacy; if only "The task is:\\n\\n", treated as empty task
 */
export async function resolveTaskBody(taskRow, runIndex = 0) {
  if (typeof taskRow.task_body === "string") {
    return taskRow.task_body;
  }

  if (taskRow.kvcomm_task && typeof taskRow.kvcomm_task === "object") {
    const spec = taskRow.kvcomm_task;
    const fixturePath = spec.fixture ?? "fixtures/kvcomm_tasks_seed42.json";
    const data = await loadFixture(fixturePath);
    const baseIndex = Number(spec.index ?? 0);
    const index = spec.advance_per_run ? baseIndex + runIndex : baseIndex;
    if (index < 0 || index >= data.tasks.length) {
      throw new Error(
        `kvcomm_task index ${index} out of range (fixture has ${data.tasks.length} tasks, task_id=${taskRow.task_id})`,
      );
    }
    return data.tasks[index];
  }

  const legacy = taskRow.user_question ?? "";
  if (legacy === "The task is:\n\n" || legacy === "The task is:\n") {
    return "";
  }
  if (legacy.startsWith("The task is:")) {
    return legacy.slice("The task is:".length).trimStart();
  }
  return legacy;
}
