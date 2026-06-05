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

function mean(values) {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, v) => sum + v, 0) / values.length;
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

/**
 * Build summary JSON from parsed bench jsonl rows. All TTFT fields use seconds (_s).
 */
export function summarizeBenchRows(rows) {
  const agentRows = rows.filter(
    (row) => row.type !== "run_summary" && typeof row.agent_index === "number",
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

  return {
    rows: rows.length,
    agent_rows: agentRows.length,
    by_agent,
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
