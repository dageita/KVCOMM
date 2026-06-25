import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");
const PROMPTS_DIR = join(BENCH_ROOT, "prompts");

const QUICK_NOTE_TASK_BODY_SUFFIX =
  / Additional stable household context for long-prefix KV bench[\s\S]*?padding is for inference benchmarking only\.$/;

/** Parse --bench-padding / BENCH_PADDING (default: off). */
export function resolveBenchPaddingEnabled(raw) {
  if (raw === undefined || raw === null || raw === "") {
    const env = process.env.BENCH_PADDING?.trim().toLowerCase();
    if (env === "1" || env === "true" || env === "yes" || env === "on") {
      return true;
    }
    if (env === "0" || env === "false" || env === "no" || env === "off") {
      return false;
    }
    return false;
  }
  if (typeof raw === "boolean") {
    return raw;
  }
  const text = String(raw).trim().toLowerCase();
  if (text === "1" || text === "true" || text === "yes" || text === "on") {
    return true;
  }
  if (text === "0" || text === "false" || text === "no" || text === "off") {
    return false;
  }
  throw new Error(`Invalid bench padding value: ${raw} (use on|off)`);
}

export function resolveRolePromptPath(benchPaddingEnabled) {
  return benchPaddingEnabled
    ? join(PROMPTS_DIR, "clawbench_chain.role.txt")
    : join(PROMPTS_DIR, "clawbench_chain.role.minimal.txt");
}

async function loadPaddingFile(relativePath) {
  const text = await readFile(join(PROMPTS_DIR, relativePath), "utf8");
  return text.trim();
}

/** User-prompt stable padding block for a task (empty when padding disabled). */
export async function resolveBenchPaddingBlock(taskRow, benchPaddingEnabled) {
  if (!benchPaddingEnabled) {
    return "";
  }
  const fromProfile = taskRow.bench_prefix_profile?.padding_file;
  if (fromProfile) {
    return loadPaddingFile(fromProfile);
  }
  if (taskRow.task_id === "t1-fs-quick-note") {
    return loadPaddingFile("t1_quick_note_stable_padding.txt");
  }
  return "";
}

/** Strip KV-bench filler from task_body when padding is off. */
export function resolveTaskBodyForBench(taskBody, taskRow, benchPaddingEnabled) {
  if (benchPaddingEnabled || typeof taskBody !== "string") {
    return taskBody;
  }
  if (taskRow.task_id === "t1-fs-quick-note") {
    return taskBody.replace(QUICK_NOTE_TASK_BODY_SUFFIX, "").trimEnd();
  }
  return taskBody;
}
