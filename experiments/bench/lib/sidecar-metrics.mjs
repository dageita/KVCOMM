/**
 * Fetch KVCOMM sidecar metrics for a bench agent record (kvcomm_latency_ms / reuse_rate).
 */

const DEFAULT_SIDECAR_URL = process.env.KVCOMM_SIDECAR_URL?.trim() || "http://127.0.0.1:8100";

function resolveRegisterTimeoutMs() {
  const raw = process.env.KVCOMM_REGISTER_TIMEOUT_MS?.trim();
  if (raw) {
    const value = Number(raw);
    if (Number.isFinite(value) && value > 0) {
      return Math.round(value);
    }
  }
  // Sidecar may block on long HF prefix prefill; default 2 min.
  return 120_000;
}

export async function registerKvcommContext(
  payload,
  { sidecarUrl = DEFAULT_SIDECAR_URL } = {},
) {
  const base = sidecarUrl.replace(/\/$/, "");
  const url = `${base}/v1/kvcomm/register`;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(resolveRegisterTimeoutMs()),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      console.warn(`[sidecar] kvcomm register failed: ${resp.status} ${text.slice(0, 200)}`);
      return false;
    }
    return true;
  } catch (err) {
    console.warn(`[sidecar] kvcomm register error: ${err?.message ?? err}`);
    return false;
  }
}

export async function fetchSidecarAgentMetrics({ runId, agentIndex, sidecarUrl = DEFAULT_SIDECAR_URL } = {}) {
  if (!runId || agentIndex == null) {
    return null;
  }
  const base = sidecarUrl.replace(/\/$/, "");
  const url = `${base}/diagnostics?run_id=${encodeURIComponent(runId)}&agent_index=${encodeURIComponent(String(agentIndex))}`;
  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) {
      return null;
    }
    const data = await resp.json();
    const row = data?.agent_metrics;
    if (!row || typeof row !== "object") {
      return null;
    }
    return {
      kvcomm_latency_ms: row.kvcomm_latency_ms ?? null,
      reuse_rate: row.reuse_rate ?? null,
      sidecar_mode: row.mode ?? null,
      sidecar_request_count: row.sidecar_request_count ?? null,
      kv_reuse_request_count: row.kv_reuse_request_count ?? null,
      dense_request_count: row.dense_request_count ?? null,
      blend_fallback_count: row.blend_fallback_count ?? null,
      sidecar_ttft_ms: row.ttft_ms ?? null,
      generation_ttft_ms: row.generation_ttft_ms ?? null,
      preprocess_latency_ms: row.preprocess_latency_ms ?? null,
      bench_no_think: row.bench_no_think ?? null,
      anchor_prediction: row.anchor_prediction ?? null,
      anchor_pooled_tokens: row.anchor_pooled_tokens ?? null,
      input_anchor_pooled_tokens: row.input_anchor_pooled_tokens ?? null,
      input_routing_mode: row.input_routing_mode ?? null,
      input_reuse_kind: row.input_reuse_kind ?? null,
      input_reuse_kinds: Array.isArray(row.input_reuse_kinds) ? row.input_reuse_kinds : null,
      reuse_kv_text: row.reuse_kv_text ?? null,
      prefix_estimated_tokens: row.prefix_estimated_tokens ?? null,
      prefix_tokens_max: row.prefix_tokens_max ?? null,
      tool_schema_tokens_sum: row.tool_schema_tokens_sum ?? null,
      response_anchor_tokens_sum: row.response_anchor_tokens_sum ?? null,
      input_anchor_tokens_sum: row.input_anchor_tokens_sum ?? null,
      response_decode_tokens_sum: row.response_decode_tokens_sum ?? null,
      short_circuit_count: row.short_circuit_count ?? null,
      emitted_tool_calls: Array.isArray(row.emitted_tool_calls) ? row.emitted_tool_calls : [],
    };
  } catch {
    return null;
  }
}

export function shouldFetchSidecarMetrics(inferenceBackend) {
  return inferenceBackend === "kvcomm_sidecar";
}
