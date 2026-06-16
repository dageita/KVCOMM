#!/usr/bin/env node
/**
 * Regenerate .report.txt from an existing bench .jsonl (+ optional .summary.json).
 *
 * Usage (from openclaw module root):
 *   node scripts/format-bench-report.mjs ../experiments/bench/results/foo.jsonl
 */

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { loadJsonl } from "../../experiments/bench/lib/load-jsonl.mjs";
import {
  summarizeBenchRows,
  summarizeClawbenchCapability,
} from "../../experiments/bench/lib/summarize-results.mjs";
import { writeBenchReport } from "../../experiments/bench/lib/bench-report.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function loadSummary(jsonlPath) {
  const rows = await loadJsonl(jsonlPath);
  const bench = summarizeBenchRows(rows);
  const clawbench = summarizeClawbenchCapability(rows);
  const summaryPath = jsonlPath.replace(/\.jsonl$/, ".summary.json");
  try {
    const raw = await readFile(summaryPath, "utf8");
    const stored = JSON.parse(raw);
    return clawbench
      ? { ...stored, ...bench, clawbench_capability: clawbench }
      : { ...stored, ...bench };
  } catch {
    return clawbench ? { ...bench, clawbench_capability: clawbench } : bench;
  }
}

async function main() {
  const input = process.argv[2];
  if (!input) {
    console.error("Usage: node scripts/format-bench-report.mjs <path/to/results.jsonl>");
    process.exit(1);
  }
  const jsonlPath = resolve(process.cwd(), input);
  const rows = await loadJsonl(jsonlPath);
  const summary = await loadSummary(jsonlPath);
  const reportPath = await writeBenchReport({ outputPath: jsonlPath, rows, summary });
  console.log("[bench-report] wrote:", reportPath);
}

main().catch((err) => {
  console.error("[bench-report] fatal:", err);
  process.exit(1);
});
