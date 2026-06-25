import { collectTtftForSession, startTtftCollection } from "./ttft-collector.mjs";
import { detectThinkingInText } from "./thinking-detect.mjs";
import {
  extractAssistantText,
  extractToolJson,
  GatewayClient,
} from "./gateway-client.mjs";
import { resolveCapabilitySubagentTools } from "./openclaw-config.mjs";
import { fetchSidecarAgentMetrics, registerKvcommContext, shouldFetchSidecarMetrics } from "./sidecar-metrics.mjs";
import { restoreImmutableCodingFiles, syncEditableCodingFiles } from "./clawbench-chain.mjs";
import { buildKvcommMetaPrefix, renderTemplateKvReuse, renderTemplateStrict, sha256Short } from "./template.mjs";

const TOOL_JSON_PATTERN =
  /\{"name"\s*:\s*"(web_search|web_fetch|update_plan|read|write|exec|process|browser|message|session_status)"/;

function isCopyTask(taskRow) {
  return taskRow?.kvcomm_ref?.domain === "COPY";
}

function analyzeOutputFormat(outputText, taskRow) {
  const text = outputText ?? "";
  const toolJsonDetected = TOOL_JSON_PATTERN.test(text);
  if (!isCopyTask(taskRow)) {
    return {
      tool_json_detected: toolJsonDetected,
      copy_char_ratio: null,
      output_format_ok: null,
    };
  }
  const copyChars = (text.match(/[ΩΔ]/g) ?? []).length;
  const copyCharRatio = text.length > 0 ? copyChars / text.length : 0;
  const outputFormatOk = !toolJsonDetected && copyCharRatio >= 0.8;
  return { tool_json_detected: toolJsonDetected, copy_char_ratio: copyCharRatio, output_format_ok: outputFormatOk };
}

function resolveAgentTasks(taskRow, spawnMode) {
  if (spawnMode === "capability" && taskRow.capability_agent_tasks) {
    return taskRow.capability_agent_tasks;
  }
  return taskRow.agent_tasks;
}

function buildSpawnParams({ taskText, model, runTimeoutSeconds, spawnMode, workspaceDir, kvcommExtraBody = null }) {
  const base = {
    task: taskText,
    mode: "run",
    cleanup: "keep",
    expectsCompletionMessage: false,
    ...(model ? { model } : {}),
    runTimeoutSeconds,
  };

  if (spawnMode === "capability") {
    return {
      ...base,
      context: "isolated",
      lightContext: false,
      ...(workspaceDir ? { cwd: workspaceDir } : {}),
      ...(kvcommExtraBody ? { extraBody: kvcommExtraBody } : {}),
    };
  }

  return {
    ...base,
    context: "isolated",
    lightContext: true,
    ...(kvcommExtraBody ? { extraBody: kvcommExtraBody } : {}),
  };
}

function renderAgentTask(template, variables, inferenceMode, agentIndex = 0) {
  const useKvReuseTemplate =
    inferenceMode === "kv_reuse" ||
    (inferenceMode === "dense_prefill" && agentIndex > 0);
  if (useKvReuseTemplate) {
    return renderTemplateKvReuse(template, variables);
  }
  return renderTemplateStrict(template, variables);
}

function buildKvcommTaskText({ taskText, runId, agentIndex, inferenceMode, messageKey, vars, taskProfile = "copy" }) {
  const meta = {
    run_id: runId,
    agent_index: agentIndex,
    node_id: String(agentIndex),
    mode: inferenceMode,
    message_key: messageKey,
    task_profile: taskProfile,
    vars,
  };
  return buildKvcommMetaPrefix(meta) + taskText;
}

function assertSpawnAccepted(result, agentIndex) {
  if (!result || result.status !== "accepted") {
    throw new Error(
      `Agent ${agentIndex} spawn failed: ${JSON.stringify(result ?? { status: "missing" })}`,
    );
  }
  if (!result.childSessionKey?.includes(":subagent:")) {
    throw new Error(
      `Agent ${agentIndex} childSessionKey missing :subagent: marker: ${JSON.stringify(result)}`,
    );
  }
}

/**
 * Stack-driven Chain spawn: tools.invoke(sessions_spawn) x N on orchestrator session.
 */
export async function runChainStackSpawn(client, params) {
  const {
    scenario,
    taskRow,
    model,
    agentId = "main",
    runTimeoutSeconds = 600,
    experimentId = "O0-pre-A",
    negativeControl = null,
    runId,
    spawnMode = "text",
    workspaceDir = null,
    inferenceMode = "dense_prefill",
    inferenceBackend = "vllm_direct",
    taskProfile = "copy",
  } = params;

  const agentCount = scenario.agent_count ?? 3;
  const agentTasks = resolveAgentTasks(taskRow, spawnMode);
  const probeAgents = new Set(scenario.ttft_probe_agents ?? [agentCount - 1]);
  const outputs = {};
  const records = [];
  const orchestratorSessionKeys = [];
  const runStartedAt = Date.now();
  const runUid = runId?.slice(0, 8) ?? "unknown";
  const capabilitySubagentTools =
    spawnMode === "capability" ? await resolveCapabilitySubagentTools() : null;

  for (let agentIndex = 0; agentIndex < agentCount; agentIndex += 1) {
    if (spawnMode === "capability" && workspaceDir) {
      await restoreImmutableCodingFiles(workspaceDir, taskRow);
    }
    const templateKey = `agent_${agentIndex}`;
    let template = agentTasks?.[templateKey];
    if (!template) {
      throw new Error(`Task ${taskRow.task_id} missing ${spawnMode === "capability" ? "capability_agent_tasks" : "agent_tasks"}.${templateKey}`);
    }

    if (negativeControl === "NC-1" && agentIndex === 1) {
      template = template.replace(/\{\{agent_0_current\}\}/g, "");
    }

    const variables = {
      user_question: taskRow.user_question ?? taskRow.task_body ?? "",
      task_body: taskRow.task_body ?? "",
      workspace_dir: workspaceDir ?? "",
      ...Object.fromEntries(
        Object.entries(outputs).map(([key, value]) => [key, value]),
      ),
    };

    let taskText = renderAgentTask(template, variables, inferenceMode, agentIndex);
    const messageKey = variables.user_question || variables.task_body || taskRow.task_id;
    const kvcommVars = {
      user_question: variables.user_question,
      task_body: variables.task_body,
      workspace_dir: variables.workspace_dir,
      ...(Array.isArray(taskRow.agent_roles)
        ? {
            agent_roles: JSON.stringify(taskRow.agent_roles),
            [`agent_${agentIndex}_role`]: taskRow.agent_roles[agentIndex] ?? "",
          }
        : {}),
      ...Object.fromEntries(
        Object.entries(outputs).map(([key, value]) => [key, value]),
      ),
    };

    const useKvcommBridge =
      inferenceBackend === "kvcomm_sidecar" &&
      (inferenceMode === "kv_reuse" || inferenceMode === "dense_prefill");
    const kvcommExtraBody = useKvcommBridge
      ? {
          kvcomm: {
            run_id: runId,
            agent_index: agentIndex,
            node_id: String(agentIndex),
            mode: inferenceMode,
            message_key: messageKey,
            task_profile: taskProfile,
            vars: kvcommVars,
          },
        }
      : null;

    if (useKvcommBridge) {
      taskText = buildKvcommTaskText({
        taskText,
        runId,
        agentIndex,
        inferenceMode,
        messageKey,
        vars: kvcommVars,
        taskProfile,
      });
      await registerKvcommContext({
        run_id: runId,
        agent_index: agentIndex,
        node_id: String(agentIndex),
        mode: inferenceMode,
        message_key: messageKey,
        vars: kvcommVars,
        task_profile: taskProfile,
        user_prompt: taskText.replace(/^<!--KVCOMM_META:\{.*?\}-->\s*/s, ""),
        system_prompt: taskRow._bench_role_prompt ?? "",
        bench_padding: Boolean(taskRow._bench_padding_enabled),
      });
    }

    const taskHash = sha256Short(taskText);

    // Fresh orchestrator session per spawn — avoids agent:main:main transcript bloat.
    const orchestratorSessionKey = await client.createSession({
      agentId,
      model: model || undefined,
      label: `kvcomm-spawn-${runUid}-a${agentIndex}`,
    });
    orchestratorSessionKeys.push(orchestratorSessionKey);

    const spawnStartedAt = Date.now();
    const invokePayload = await client.invokeTool(
      orchestratorSessionKey,
      "sessions_spawn",
      buildSpawnParams({
        taskText,
        model,
        runTimeoutSeconds,
        spawnMode,
        workspaceDir,
        kvcommExtraBody,
      }),
    );

    const spawnResult = extractToolJson(invokePayload);
    assertSpawnAccepted(spawnResult, agentIndex);

    const childSessionKey = spawnResult.childSessionKey;
    let childRunId = spawnResult.runId;
    let runStartedAt = spawnStartedAt;
    let ttftPromise;

    if (spawnMode === "capability" && capabilitySubagentTools?.length && workspaceDir) {
      await client.patchSession(childSessionKey, {
        inheritedToolAllow: capabilitySubagentTools,
        inheritedToolDeny: null,
        spawnedWorkspaceDir: workspaceDir,
      });
      if (childRunId) {
        await client.abortSession(childSessionKey, childRunId).catch(() => null);
      }
      await client.resetSession(childSessionKey, { reason: "reset" });
      await client.patchSession(childSessionKey, {
        inheritedToolAllow: capabilitySubagentTools,
        inheritedToolDeny: null,
        spawnedWorkspaceDir: workspaceDir,
      });
      const sent = await client.sendSession(childSessionKey, taskText, {
        timeoutMs: runTimeoutSeconds * 1000,
      });
      childRunId = sent.runId;
      runStartedAt = sent.startedAt;
      const ttftDeadlineMs = runTimeoutSeconds * 1000;
      ttftPromise = startTtftCollection(client, {
        sessionKey: childSessionKey,
        runId: childRunId,
        sinceMs: runStartedAt,
        untilMs: runStartedAt + ttftDeadlineMs + 15_000,
        timeoutMs: ttftDeadlineMs + 15_000,
      });
      await client.agentWait(childRunId, runTimeoutSeconds * 1000);
    } else {
      const ttftDeadlineMs = runTimeoutSeconds * 1000;
      ttftPromise = startTtftCollection(client, {
        sessionKey: childSessionKey,
        runId: childRunId,
        sinceMs: runStartedAt,
        untilMs: runStartedAt + ttftDeadlineMs + 15_000,
        timeoutMs: ttftDeadlineMs + 15_000,
      });
      if (childRunId) {
        await client.agentWait(childRunId, runTimeoutSeconds * 1000);
      } else {
        await client.waitForTask(childSessionKey, {
          timeoutMs: runTimeoutSeconds * 1000,
        });
      }
    }

    const messages = await client.getSessionMessages(childSessionKey);
    const outputText = extractAssistantText(messages);

    let sidecarMetrics = null;
    if (shouldFetchSidecarMetrics(inferenceBackend)) {
      sidecarMetrics = await fetchSidecarAgentMetrics({ runId, agentIndex });
    }

    const ttftInfo = await ttftPromise;
    const ttftFallbackSince = spawnMode === "capability" ? runStartedAt : spawnStartedAt;
    let resolvedTtft =
      ttftInfo?.ttft_gateway_assistant_ms != null || ttftInfo?.ttft_ms != null
        ? ttftInfo
        : await collectTtftForSession(childSessionKey, {
            sinceMs: ttftFallbackSince,
            untilMs: Date.now(),
            wallClockMs: Date.now() - ttftFallbackSince,
            runId: childRunId,
            client,
          });

    if (
      (resolvedTtft?.ttft_fallback || resolvedTtft?.ttft_gateway_assistant_ms == null) &&
      sidecarMetrics?.sidecar_ttft_ms != null
    ) {
      resolvedTtft = {
        ...resolvedTtft,
        ttft_ms: sidecarMetrics.sidecar_ttft_ms,
        ttft_gateway_assistant_ms: null,
        ttft_source: "sidecar.inference",
        ttft_fallback: true,
        ttft_note:
          "No WS stream observed; using sidecar preprocess+generation_ttft (inference-only, not gateway user-perceived TTFT).",
      };
    }

    outputs[`agent_${agentIndex}_current`] = outputText;
    const outputFormat = analyzeOutputFormat(outputText, taskRow);

    const upstream_hashes = {};
    if (agentIndex > 0) {
      for (let j = 0; j < agentIndex; j += 1) {
        const key = `agent_${j}_current`;
        const value = outputs[key];
        upstream_hashes[key] = value ? sha256Short(value) : null;
      }
    }

    let taskIncludesUpstream = true;
    if (agentIndex > 0) {
      if (scenario.topology === "chain") {
        const predKey = `agent_${agentIndex - 1}_current`;
        taskIncludesUpstream = taskText.includes(outputs[predKey] ?? "__missing__");
      } else {
        taskIncludesUpstream = Object.keys(upstream_hashes).every((key) =>
          taskText.includes(outputs[key] ?? "__missing__"),
        );
      }
    }

    const record = {
      experiment_id: experimentId,
      task_id: taskRow.task_id,
      run_id: params.runId,
      agent_index: agentIndex,
      node_id: String(agentIndex),
      probe: probeAgents.has(agentIndex),
      child_session_key: childSessionKey,
      child_run_id: childRunId ?? null,
      orchestrator_session_key: orchestratorSessionKey,
      spawn_mode: spawnMode,
      workspace_dir: workspaceDir,
      task_hash: taskHash,
      upstream_hashes,
      task_includes_upstream: taskIncludesUpstream,
      output_text: outputText,
      output_len: outputText.length,
      ...outputFormat,
      ttft_ms: resolvedTtft.ttft_ms,
      ttft_gateway_assistant_ms: resolvedTtft.ttft_gateway_assistant_ms ?? resolvedTtft.ttft_ms ?? null,
      ttft_gateway_thinking_ms: resolvedTtft.ttft_gateway_thinking_ms ?? null,
      ttft_thinking_to_assistant_ms: resolvedTtft.ttft_thinking_to_assistant_ms ?? null,
      ttft_source: resolvedTtft.source,
      ttft_fallback: resolvedTtft.fallback,
      ttft_note: resolvedTtft.note ?? null,
      thinking_detected: detectThinkingInText(outputText),
      generation_ttft_ms: sidecarMetrics?.generation_ttft_ms ?? null,
      preprocess_latency_ms: sidecarMetrics?.preprocess_latency_ms ?? null,
      sidecar_ttft_ms: sidecarMetrics?.sidecar_ttft_ms ?? null,
      kvcomm_latency_ms: sidecarMetrics?.kvcomm_latency_ms ?? null,
      reuse_rate: sidecarMetrics?.reuse_rate ?? null,
      sidecar_mode: sidecarMetrics?.sidecar_mode ?? null,
      anchor_prediction: sidecarMetrics?.anchor_prediction ?? null,
      anchor_pooled_tokens: sidecarMetrics?.anchor_pooled_tokens ?? null,
      input_anchor_pooled_tokens: sidecarMetrics?.input_anchor_pooled_tokens ?? null,
      input_routing_mode: sidecarMetrics?.input_routing_mode ?? null,
      reuse_kv_text: sidecarMetrics?.reuse_kv_text ?? null,
      prefix_estimated_tokens: sidecarMetrics?.prefix_estimated_tokens ?? null,
      bench_no_think: sidecarMetrics?.bench_no_think ?? true,
      e2e_agent_ms: Date.now() - spawnStartedAt,
      timestamp: new Date().toISOString(),
    };

    records.push(record);

    if (spawnMode === "capability" && workspaceDir) {
      await syncEditableCodingFiles(workspaceDir, taskRow);
      await restoreImmutableCodingFiles(workspaceDir, taskRow);
    }
  }

  return {
    experiment_id: experimentId,
    task_id: taskRow.task_id,
    run_id: params.runId,
    agent_count: agentCount,
    orchestrator_session_keys: orchestratorSessionKeys,
    e2e_run_ms: Date.now() - runStartedAt,
    records,
    outputs,
    spawn_mode: spawnMode,
    workspace_dir: workspaceDir,
  };
}

export async function connectGateway(options) {
  const client = await GatewayClient.create(options);
  await client.connect();
  return client;
}
