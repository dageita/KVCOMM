import { writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

import { msToSeconds } from "./summarize-results.mjs";

const LINE = "─".repeat(72);
const DOUBLE = "═".repeat(72);

function pad(value, width, align = "left") {
  const text = value == null ? "" : String(value);
  if (text.length >= width) {
    return text.slice(0, width);
  }
  const padLen = width - text.length;
  return align === "right" ? " ".repeat(padLen) + text : text + " ".repeat(padLen);
}

function fmtNum(value, digits = 3) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function fmtReuseText(text, maxLen = 40) {
  if (!text) {
    return "n/a";
  }
  const compact = String(text).replace(/\s+/g, " ").trim();
  if (compact.length <= maxLen) {
    return compact;
  }
  return `${compact.slice(0, maxLen - 3)}...`;
}

function fmtMs(value) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return `${Math.round(value)}`;
}

function fmtMeasureKvReuseRate(summary) {
  if (summary?.measure_kv_reuse_rate == null) {
    return null;
  }
  const count = summary.measure_kv_reuse_count ?? 0;
  const total = summary.measure_sidecar_requests ?? 0;
  const pct = Math.round(summary.measure_kv_reuse_rate * 100);
  return `${count}/${total} (${pct}%)`;
}

function firstMeta(rows) {
  const agent = rows.find((row) => typeof row.agent_index === "number");
  const summary = rows.find((row) => row.type === "run_summary");
  return agent ?? summary ?? rows[0] ?? {};
}

function groupRunsInOrder(rows) {
  const runs = [];
  const seen = new Set();
  for (const row of rows) {
    const runId = row.run_id;
    if (!runId || seen.has(runId)) {
      continue;
    }
    seen.add(runId);
    const agents = rows
      .filter((r) => r.run_id === runId && typeof r.agent_index === "number")
      .sort((a, b) => a.agent_index - b.agent_index);
    const summary = rows.find((r) => r.run_id === runId && r.type === "run_summary") ?? null;
    runs.push({
      run_id: runId,
      run_uid: row.run_uid ?? runId.slice(0, 8),
      warmup: row.warmup === true,
      agents,
      summary,
    });
  }
  return runs;
}

function sidecarRoutingTable(agentRows) {
  const byAgent = new Map();
  for (const row of agentRows) {
    const key = String(row.agent_index);
    if (!byAgent.has(key)) {
      byAgent.set(key, { kv: 0, dense: 0, reuse: [], blendFallbacks: 0 });
    }
    const bucket = byAgent.get(key);
    const totalReq = row.sidecar_request_count ?? 1;
    const kvReq =
      row.kv_reuse_request_count ??
      (row.sidecar_mode === "kv_reuse" ? totalReq : 0);
    const denseReq =
      row.dense_request_count ??
      (row.sidecar_mode === "dense_prefill" ? totalReq : totalReq - kvReq);
    bucket.kv += kvReq;
    bucket.dense += denseReq;
    if (typeof row.reuse_rate === "number") {
      bucket.reuse.push(row.reuse_rate);
    }
    if (typeof row.blend_fallback_count === "number") {
      bucket.blendFallbacks += row.blend_fallback_count;
    }
  }

  const lines = [
    "Agent   │ kv_reuse   │ dense      │ reuse avg   ",
    "────────┼────────────┼────────────┼─────────────",
  ];
  for (const [agentIndex, bucket] of [...byAgent.entries()].sort((a, b) => Number(a[0]) - Number(b[0]))) {
    const total = bucket.kv + bucket.dense;
    const reuseAvg =
      bucket.reuse.length === 0
        ? "n/a"
        : fmtNum(bucket.reuse.reduce((s, v) => s + v, 0) / bucket.reuse.length, 2);
    const blendNote =
      bucket.blendFallbacks > 0 ? ` (${bucket.blendFallbacks} blend fallbacks)` : "";
    lines.push(
      `${pad(agentIndex, 7)} │ ${pad(`${bucket.kv}/${total}`, 10)} │ ${pad(`${bucket.dense}/${total}`, 10)} │ ${pad(reuseAvg, 11)}${blendNote}`,
    );
  }
  return lines.join("\n");
}

function inferenceTtftTable(summary) {
  const byAgent = summary?.inference_by_agent ?? {};
  const agentIndices = Object.keys(byAgent).sort((a, b) => Number(a) - Number(b));
  if (agentIndices.length === 0) {
    return "  (no inference timing samples)";
  }
  const lines = [
    "Agent   │ gen_ttft   │ kvcomm_ms  │ sidecar_ttft │ Samples ",
    "        │ avg (ms)   │ avg (ms)   │ avg (ms)     │         ",
    "────────┼────────────┼────────────┼──────────────┼─────────",
  ];
  for (const agentIndex of agentIndices) {
    const stats = byAgent[agentIndex];
    const samples = Math.max(
      stats.gen_ttft?.samples ?? 0,
      stats.kvcomm_ms?.samples ?? 0,
      stats.sidecar_ttft_ms?.samples ?? 0,
    );
    lines.push(
      `${pad(agentIndex, 7)} │ ${pad(fmtMs(stats.gen_ttft?.avg_ms), 10)} │ ${pad(fmtMs(stats.kvcomm_ms?.avg_ms), 10)} │ ${pad(fmtMs(stats.sidecar_ttft_ms?.avg_ms), 12)} │ ${pad(samples, 7)}`,
    );
  }
  lines.push("");
  lines.push(
    "  gen_ttft = HF 首 decode token；kvcomm_ms = KV 预处理；sidecar_ttft = preprocess + decode 总 TTFT",
  );
  lines.push(
    "  对比 kv_reuse vs dense_prefill 时请分别跑两次 bench，对照本表（勿用 asst(s)，含 OpenClaw 编排开销）",
  );
  return lines.join("\n");
}

function ttftByAgentTable(summary) {
  const lines = [
    "Agent   │ Samples  │ Avg (s)    │ P50 (s)    │ P99 (s)    │ Probe ",
    "────────┼──────────┼────────────┼────────────┼────────────┼───────",
  ];
  const byAgent = summary?.by_agent ?? {};
  for (const agentIndex of Object.keys(byAgent).sort((a, b) => Number(a) - Number(b))) {
    const stats = byAgent[agentIndex];
    lines.push(
      `${pad(agentIndex, 7)} │ ${pad(stats.samples, 8)} │ ${pad(fmtNum(stats.ttft_avg_s), 10)} │ ${pad(fmtNum(stats.ttft_p50_s), 10)} │ ${pad(fmtNum(stats.ttft_p99_s), 10)} │ ${pad(stats.probe ? "yes" : "", 5)}`,
    );
  }
  if (summary?.probe?.samples) {
    const p = summary.probe;
    lines.push("");
    lines.push(
      ` Probe agent: avg=${fmtNum(p.ttft_avg_s)}s p50=${fmtNum(p.ttft_p50_s)}s p99=${fmtNum(p.ttft_p99_s)}s (n=${p.samples})`,
    );
  }
  return lines.join("\n");
}

function capabilityTable(summary) {
  const cap = summary?.clawbench_capability;
  if (!cap?.by_measure?.length) {
    return null;
  }
  const lines = [
    "#    │ run_uid    │ Run     │ C       │ T       │ B       │ Status  ",
    "─────┼────────────┼─────────┼─────────┼─────────┼─────────┼─────────",
  ];
  for (const row of cap.by_measure) {
    const status = row.passed ? "PASS" : (row.run_score ?? 0) >= 0.4 ? "PARTIAL" : "FAIL";
    lines.push(
      `${pad(row.measure_index, 4)} │ ${pad(row.run_uid, 10)} │ ${pad(fmtNum(row.run_score, 1), 7)} │ ${pad(fmtNum(row.completion_score, 1), 7)} │ ${pad(fmtNum(row.trajectory_score, 1), 7)} │ ${pad(fmtNum(row.behavior_score, 1), 7)} │ ${pad(status, 7)}`,
    );
  }
  lines.push("");
  lines.push(
    ` Aggregate (${cap.runs} run(s)): run=${fmtNum(cap.run_score_avg, 1)} C=${fmtNum(cap.completion_score_avg, 1)} T=${fmtNum(cap.trajectory_score_avg, 1)} B=${fmtNum(cap.behavior_score_avg, 1)} pass_rate=${Math.round((cap.pass_rate ?? 0) * 100)}%`,
  );
  return lines.join("\n");
}

function perRunDetailSection(runs) {
  const lines = [
    `${LINE}`,
    " Per-run / per-agent detail",
    `${LINE}`,
    "  asst(s)     = Gateway WS 首条可见 assistant delta（用户可感知 TTFT）",
    "  gen_ttft    = HF sidecar 首个 decode token 延迟（ms）",
    "  sidecar_ttft= sidecar 推理总 TTFT：preprocess + decode（ms）",
    "  Mode        = sidecar 实际推理路径（kv_reuse / dense_prefill）",
    "  reuse       = 本请求是否走 kv_reuse（1.0/0.0）",
    "  prefix      = OpenClaw prefix 估算 token 数",
    "  resp_anch   = 本次新入池的 response anchor token 数（0=已有 anchor，未重复入池）",
    "  in_anch     = 本次新入池的 input anchor token 数（0=复用已有 input KV）",
    "  reuse_txt   = kv_reuse 路径复用的 placeholder KV 解码文本（截断）",
    "  kvcomm_ms   = sidecar KV 预处理耗时（ms）",
    "  e2e(s)      = 单 agent spawn 端到端耗时（s）",
    "  measure_kv_reuse_rate = 本 measure 中 agent 请求走 kv_reuse 的比例",
    "",
  ];

  let measureIndex = 0;
  for (let runIndex = 0; runIndex < runs.length; runIndex += 1) {
    const run = runs[runIndex];
    const kind = run.warmup ? "warmup" : "measure";
    if (!run.warmup) {
      measureIndex += 1;
    }
    const label = run.warmup
      ? `Run ${runIndex + 1} [warmup]  run_uid=${run.run_uid}`
      : `Run ${runIndex + 1} [measure #${measureIndex}]  run_uid=${run.run_uid}`;

    lines.push(label);
    if (run.summary?.e2e_run_ms != null) {
      lines.push(`  e2e_run_ms=${run.summary.e2e_run_ms}`);
    }
    if (run.summary?.capability_score && !run.summary.capability_score.error) {
      const s = run.summary.capability_score;
      lines.push(
        `  capability: run=${fmtNum(s.run_score, 1)} C=${fmtNum(s.completion_score, 1)} T=${fmtNum(s.trajectory_score, 1)} B=${fmtNum(s.behavior_score, 1)}`,
      );
    }
    const measureReuse = fmtMeasureKvReuseRate(run.summary);
    if (!run.warmup && measureReuse) {
      lines.push(`  measure_kv_reuse_rate=${measureReuse}`);
    }

    lines.push(
      "  Agent │ asst(s) │ gen_ttft │ sidecar_ttft │ Mode           │ reuse │ prefix │ resp_anch │ in_anch │ reuse_txt                            │ kvcomm_ms │ e2e(s)",
    );
    lines.push(
      "  ──────┼─────────┼──────────┼──────────────┼────────────────┼───────┼────────┼─────────┼─────────┼──────────────────────────────────────┼───────────┼────────",
    );

    for (const agent of run.agents) {
      lines.push(
        `  ${pad(agent.agent_index, 5)} │ ${pad(fmtNum(msToSeconds(agent.ttft_gateway_assistant_ms ?? agent.ttft_ms)), 7)} │ ${pad(fmtMs(agent.generation_ttft_ms), 8)} │ ${pad(fmtMs(agent.sidecar_ttft_ms), 12)} │ ${pad(agent.sidecar_mode ?? "n/a", 14)} │ ${pad(agent.reuse_rate == null ? "n/a" : fmtNum(agent.reuse_rate, 2), 5)} │ ${pad(agent.prefix_estimated_tokens == null ? "n/a" : String(agent.prefix_estimated_tokens), 6)} │ ${pad(agent.anchor_pooled_tokens == null ? "n/a" : String(agent.anchor_pooled_tokens), 7)} │ ${pad(agent.input_anchor_pooled_tokens == null ? "n/a" : String(agent.input_anchor_pooled_tokens), 7)} │ ${pad(fmtReuseText(agent.reuse_kv_text), 36)} │ ${pad(fmtMs(agent.kvcomm_latency_ms), 9)} │ ${pad(fmtNum(msToSeconds(agent.e2e_agent_ms)), 6)}`,
      );
    }
    lines.push("");
  }
  return lines.join("\n");
}

/**
 * Build human-readable bench report (aggregate + per-run/per-agent detail).
 */
export function formatBenchReport(rows, summary) {
  const meta = firstMeta(rows);
  const measureAgentRows = rows.filter(
    (row) => row.type !== "run_summary" && typeof row.agent_index === "number" && row.warmup !== true,
  );
  const allRuns = groupRunsInOrder(rows);
  const warmupRuns = allRuns.filter((run) => run.warmup).length;
  const measureRuns = allRuns.length - warmupRuns;

  const parts = [
    DOUBLE,
    " BENCH RUN SUMMARY",
    DOUBLE,
    "",
    `  Experiment:    ${meta.experiment_id ?? "n/a"}`,
    `  Task:          ${meta.task_id ?? "n/a"}`,
    `  Model:         ${meta.model ?? "n/a"}`,
    `  Inference:     ${meta.inference_mode ?? "n/a"} / ${meta.inference_backend ?? "n/a"}`,
    `  Profile:       ${meta.task_profile ?? "n/a"}`,
    `  Workload:      ${meta.workload ?? "n/a"}`,
    `  Agents:        ${meta.agent_count ?? "n/a"}`,
    `  Spawn:         ${meta.spawn_mode ?? "n/a"}`,
    `  Runs:          ${allRuns.length} total (${warmupRuns} warmup, ${measureRuns} measure)`,
    "",
    LINE,
    " Overview (measure runs only)",
    LINE,
    `  Rows: ${summary?.rows ?? rows.length}  |  Agent rows (measure): ${measureAgentRows.length}  |  comms_ok: ${summary?.comms_ok_rate != null ? `${Math.round(summary.comms_ok_rate * 100)}%` : "n/a"}`,
  ];
  if (summary?.measure_kv_reuse_rate_avg != null) {
    parts.push(
      `  Measure kv_reuse rate (avg across runs): ${fmtNum(summary.measure_kv_reuse_rate_avg * 100, 1)}% (${summary.measure_kv_reuse_runs ?? 0} run(s))`,
    );
  }
  parts.push(
    "",
    LINE,
    " Inference TTFT (measure only, sidecar/HF)",
    LINE,
    inferenceTtftTable(summary),
    "",
    LINE,
    " Gateway TTFT by agent (seconds, measure only — includes orchestration)",
    LINE,
    ttftByAgentTable(summary),
    "",
    LINE,
    " Sidecar routing (measure only)",
    LINE,
    sidecarRoutingTable(measureAgentRows),
  );

  const capTable = capabilityTable(summary);
  if (capTable) {
    parts.push(
      "",
      LINE,
      " ClawBench capability (0–1, pass ≥ 0.7)",
      LINE,
      capTable,
    );
  }

  parts.push("", perRunDetailSection(allRuns), DOUBLE);
  return `${parts.join("\n")}\n`;
}

export function reportPathForOutput(outputPath) {
  return outputPath.endsWith(".jsonl")
    ? outputPath.replace(/\.jsonl$/, ".report.txt")
    : `${outputPath}.report.txt`;
}

export async function writeBenchReport({ outputPath, rows, summary }) {
  const reportPath = reportPathForOutput(outputPath);
  const text = formatBenchReport(rows, summary);
  await writeFile(reportPath, text, "utf8");
  return reportPath;
}

export async function writeBenchReportFromFile(outputPath, summary) {
  const { readFile } = await import("node:fs/promises");
  const raw = await readFile(outputPath, "utf8");
  const rows = raw
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return writeBenchReport({ outputPath, rows, summary });
}
