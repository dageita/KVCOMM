function roundSeconds(value) {
  if (value == null || Number.isNaN(value)) {
    return null;
  }
  return Math.round(value * 1000) / 1000;
}

export function msToSeconds(ms) {
  if (ms == null || typeof ms !== "number") {
    return null;
  }
  return roundSeconds(ms / 1000);
}

function percentile(values, p) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

function ttftStatsFromMs(valuesMs) {
  const seconds = valuesMs.map((ms) => ms / 1000);
  return {
    samples: valuesMs.length,
    ttft_avg_s: roundSeconds(mean(seconds)),
    ttft_p50_s: roundSeconds(percentile(seconds, 50)),
    ttft_p99_s: roundSeconds(percentile(seconds, 99)),
  };
}

function msStats(valuesMs) {
  if (valuesMs.length === 0) {
    return { samples: 0, avg_ms: null, p50_ms: null, p99_ms: null };
  }
  return {
    samples: valuesMs.length,
    avg_ms: Math.round(mean(valuesMs) * 100) / 100,
    p50_ms: Math.round(percentile(valuesMs, 50) * 100) / 100,
    p99_ms: Math.round(percentile(valuesMs, 99) * 100) / 100,
  };
}

function mean(values) {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Aggregate ClawBench native scores from run_summary rows (score_task_run output).
 */
function isMeasureRow(row) {
  return row.warmup !== true;
}

export function summarizeClawbenchCapability(rows) {
  const summaries = rows.filter(
    (row) => row.type === "run_summary" && row.capability_score && isMeasureRow(row),
  );
  if (summaries.length === 0) {
    return null;
  }

  const scored = summaries.filter((row) => !row.capability_score.error);
  const round3 = (v) => (v == null ? null : Math.round(v * 1000) / 1000);

  return {
    runs: summaries.length,
    scored_runs: scored.length,
    run_score_avg: round3(mean(scored.map((row) => row.capability_score.run_score))),
    completion_score_avg: round3(mean(scored.map((row) => row.capability_score.completion_score))),
    trajectory_score_avg: round3(mean(scored.map((row) => row.capability_score.trajectory_score))),
    behavior_score_avg: round3(mean(scored.map((row) => row.capability_score.behavior_score))),
    pass_rate:
      scored.length === 0
        ? 0
        : scored.filter((row) => (row.capability_score.run_score ?? 0) >= 0.7).length /
          scored.length,
    by_measure: summaries.map((row, index) => {
      const score = row.capability_score;
      if (score.error) {
        return {
          measure_index: index + 1,
          task_id: row.task_id,
          run_id: row.run_id,
          run_uid: row.run_uid,
          error: score.error,
        };
      }
      return {
        measure_index: index + 1,
        task_id: row.task_id,
        run_id: row.run_id,
        run_uid: row.run_uid,
        run_score: round3(score.run_score),
        completion_score: round3(score.completion_score),
        trajectory_score: round3(score.trajectory_score),
        behavior_score: round3(score.behavior_score),
        judge_score: round3(score.judge_score),
        measure_kv_reuse_rate: row.measure_kv_reuse_rate ?? null,
        measure_kv_reuse_count: row.measure_kv_reuse_count ?? null,
        measure_sidecar_requests: row.measure_sidecar_requests ?? null,
        passed: (score.run_score ?? 0) >= 0.7,
        failed_assertions: score.failed_assertions ?? [],
        trajectory_violations: score.trajectory_violations ?? [],
      };
    }),
    units: {
      scores: "0-1 (ClawBench score_task_run)",
      pass_threshold: 0.7,
    },
  };
}

/**
 * Build summary JSON from parsed bench jsonl rows. All TTFT fields use seconds (_s).
 */
export function summarizeBenchRows(rows) {
  const agentRows = rows.filter(
    (row) =>
      row.type !== "run_summary" && typeof row.agent_index === "number" && isMeasureRow(row),
  );
  const withTtft = agentRows.filter((row) => typeof row.ttft_ms === "number");

  const byAgent = {};
  for (const row of withTtft) {
    const key = String(row.agent_index);
    if (!byAgent[key]) {
      byAgent[key] = [];
    }
    byAgent[key].push(row.ttft_ms);
  }

  const by_agent = {};
  for (const [agentIndex, ttftMsList] of Object.entries(byAgent)) {
    const agentOnly = agentRows.filter((r) => String(r.agent_index) === agentIndex);
    const stats = ttftStatsFromMs(ttftMsList);
    by_agent[agentIndex] = {
      ...stats,
      probe: agentOnly.some((r) => r.probe === true),
      ttft_fallback_rate:
        agentOnly.length === 0
          ? 1
          : agentOnly.filter((r) => r.ttft_fallback).length / agentOnly.length,
      output_format_ok_rate:
        agentOnly.length === 0
          ? 0
          : agentOnly.filter((r) => r.output_format_ok).length / agentOnly.length,
    };
  }

  const probeRows = withTtft.filter((row) => row.probe === true);
  const probeTtftMs = probeRows.map((row) => row.ttft_ms);
  const measureSummaries = rows.filter(
    (row) => row.type === "run_summary" && isMeasureRow(row),
  );
  const measureReuseRates = measureSummaries
    .map((row) => row.measure_kv_reuse_rate)
    .filter((value) => typeof value === "number");

  const inference_by_agent = {};
  for (const row of agentRows) {
    const key = String(row.agent_index);
    if (!inference_by_agent[key]) {
      inference_by_agent[key] = {
        gen_ttft_ms: [],
        kvcomm_ms: [],
        sidecar_ttft_ms: [],
      };
    }
    const bucket = inference_by_agent[key];
    if (typeof row.generation_ttft_ms === "number") {
      bucket.gen_ttft_ms.push(row.generation_ttft_ms);
    }
    if (typeof row.kvcomm_latency_ms === "number") {
      bucket.kvcomm_ms.push(row.kvcomm_latency_ms);
    }
    if (typeof row.sidecar_ttft_ms === "number") {
      bucket.sidecar_ttft_ms.push(row.sidecar_ttft_ms);
    }
  }
  const inference_by_agent_summary = {};
  for (const [agentIndex, bucket] of Object.entries(inference_by_agent)) {
    inference_by_agent_summary[agentIndex] = {
      gen_ttft: msStats(bucket.gen_ttft_ms),
      kvcomm_ms: msStats(bucket.kvcomm_ms),
      sidecar_ttft_ms: msStats(bucket.sidecar_ttft_ms),
    };
  }

  return {
    rows: rows.length,
    agent_rows: agentRows.length,
    by_agent,
    inference_by_agent: inference_by_agent_summary,
    measure_kv_reuse_rate_avg:
      measureReuseRates.length === 0 ? null : roundSeconds(mean(measureReuseRates)),
    measure_kv_reuse_runs: measureReuseRates.length,
    probe: {
      ...ttftStatsFromMs(probeTtftMs),
      ttft_fallback_rate:
        probeRows.length === 0
          ? 1
          : probeRows.filter((row) => row.ttft_fallback).length / probeRows.length,
    },
    comms_ok_rate:
      agentRows.length === 0
        ? 0
        : agentRows.filter((row) => row.task_includes_upstream !== false).length /
          agentRows.length,
    units: { ttft: "s" },
  };
}
