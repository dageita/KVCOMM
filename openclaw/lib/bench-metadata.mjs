/**
 * Sidecar-ready benchmark metadata (harness / inference backend / mode).
 * Driver attaches these fields to every jsonl row; sidecar metrics may be null until bridge lands.
 */

export const TTFT_DEFINITION_CAPABILITY =
  "sessions_send_to_first_assistant_delta_gateway_ws";

export const TTFT_DEFINITION_TEXT_SPAWN =
  "spawn_accepted_to_first_assistant_delta_gateway_ws";

/** @deprecated use TTFT_DEFINITION_CAPABILITY / TTFT_DEFINITION_TEXT_SPAWN */
export const TTFT_DEFINITION = TTFT_DEFINITION_CAPABILITY;

export const HARNESS = "openclaw";

export function resolveInferenceBackend(env = process.env, explicit) {
  if (explicit) {
    return explicit;
  }
  if (env.KVCOMM_SIDECAR_URL?.trim() || env.KVCOMM_INFERENCE_BACKEND === "kvcomm_sidecar") {
    return "kvcomm_sidecar";
  }
  return "vllm_direct";
}

export function resolveInferenceMode(env = process.env, explicit) {
  if (explicit) {
    return explicit;
  }
  const mode = env.KVCOMM_MODE?.trim();
  if (mode === "kv_reuse" || mode === "dense_prefill") {
    return mode;
  }
  return "dense_prefill";
}

export function resolveWorkload(taskProfile = "copy") {
  if (taskProfile === "clawbench") {
    return "clawbench_chain";
  }
  return "copy_chain";
}

export function buildRunMetadata(options = {}) {
  const {
    experimentId = "O0-pre-A",
    scenario = {},
    model = "",
    inferenceMode,
    inferenceBackend,
    warmup = false,
    taskProfile = "copy",
    copyPrefixRepeats = Number(process.env.COPY_PREFIX_REPEATS || "64"),
    copyOutLength = Number(process.env.COPY_OUT_LENGTH || "128"),
    sidecarVersion = process.env.KVCOMM_SIDECAR_VERSION?.trim() || null,
    spawnMode = "text",
  } = options;

  const agentCount = scenario.agent_count ?? 3;
  const inference_backend = resolveInferenceBackend(process.env, inferenceBackend);
  const inference_mode = resolveInferenceMode(process.env, inferenceMode);
  const workload = resolveWorkload(taskProfile);

  const notComparable = ["ttft_absolute_s"];
  if (taskProfile === "copy") {
    notComparable.push("output_format_ok");
  }
  if (taskProfile === "clawbench") {
    notComparable.push("clawbench_judge_score", "clawbench_completion_score");
  }

  return {
    harness: HARNESS,
    harness_module: "kvcomm/openclaw",
    experiment_id: experimentId,
    topology: scenario.topology ?? "chain",
    agent_count: agentCount,
    ttft_probe_agents: scenario.ttft_probe_agents ?? [agentCount - 1],
    workload,
    task_profile: taskProfile,
    spawn_mode: spawnMode,
    copy_prefix_repeats: taskProfile === "copy" ? copyPrefixRepeats : null,
    copy_out_length: taskProfile === "copy" ? copyOutLength : null,
    model: model || null,
    inference_backend,
    inference_mode,
    sidecar_version: sidecarVersion,
    sidecar_url: process.env.KVCOMM_SIDECAR_URL?.trim() || null,
    warmup,
    ttft_definition:
      spawnMode === "capability" ? TTFT_DEFINITION_CAPABILITY : TTFT_DEFINITION_TEXT_SPAWN,
    bench_no_think: process.env.KVCOMM_BENCH_NO_THINK?.trim()
      ? !["0", "false", "no", "off"].includes(process.env.KVCOMM_BENCH_NO_THINK.trim().toLowerCase())
      : true,
    comparable_to: `kvcomm_python_${scenario.topology ?? "chain"}_${agentCount}`,
    not_comparable_fields: notComparable,
  };
}

export function computeRunKvReuseStats(records = []) {
  const routed = records.filter((record) => record.sidecar_mode);
  if (routed.length === 0) {
    return {
      measure_kv_reuse_rate: null,
      measure_kv_reuse_count: null,
      measure_sidecar_requests: null,
    };
  }
  const kvCount = routed.filter((record) => record.sidecar_mode === "kv_reuse").length;
  return {
    measure_kv_reuse_rate: kvCount / routed.length,
    measure_kv_reuse_count: kvCount,
    measure_sidecar_requests: routed.length,
  };
}

export function enrichAgentRecord(record, meta, runUid) {
  return {
    ...record,
    run_uid: runUid.slice(0, 8),
    ...meta,
    ttft_gateway_assistant_ms: record.ttft_gateway_assistant_ms ?? record.ttft_ms ?? null,
    ttft_gateway_thinking_ms: record.ttft_gateway_thinking_ms ?? null,
    ttft_thinking_to_assistant_ms: record.ttft_thinking_to_assistant_ms ?? null,
    thinking_detected: record.thinking_detected ?? null,
    generation_ttft_ms: record.generation_ttft_ms ?? null,
    preprocess_latency_ms: record.preprocess_latency_ms ?? null,
    sidecar_ttft_ms: record.sidecar_ttft_ms ?? null,
    kvcomm_latency_ms: record.kvcomm_latency_ms ?? null,
    reuse_rate: record.reuse_rate ?? null,
    anchor_prediction: record.anchor_prediction ?? null,
    anchor_pooled_tokens: record.anchor_pooled_tokens ?? null,
    input_anchor_pooled_tokens: record.input_anchor_pooled_tokens ?? null,
    input_routing_mode: record.input_routing_mode ?? null,
    reuse_kv_text: record.reuse_kv_text ?? null,
    prefix_estimated_tokens: record.prefix_estimated_tokens ?? null,
  };
}

export function enrichRunSummary(summary, meta, runUid) {
  return {
    ...summary,
    run_uid: runUid.slice(0, 8),
    ...meta,
  };
}
