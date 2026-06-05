import { collectTtftForSession, startTtftCollection } from "./ttft-collector.mjs";
import {
  extractAssistantText,
  extractToolJson,
  GatewayClient,
} from "./gateway-client.mjs";
import { renderTemplateStrict, sha256Short } from "./template.mjs";

const TOOL_JSON_PATTERN =
  /\{"name"\s*:\s*"(web_search|web_fetch|update_plan|read|write|exec|process|browser|message|session_status)"/;

function analyzeOutputFormat(outputText) {
  const text = outputText ?? "";
  const toolJsonDetected = TOOL_JSON_PATTERN.test(text);
  const copyChars = (text.match(/[ΩΔ]/g) ?? []).length;
  const copyCharRatio = text.length > 0 ? copyChars / text.length : 0;
  const outputFormatOk = !toolJsonDetected && copyCharRatio >= 0.8;
  return { tool_json_detected: toolJsonDetected, copy_char_ratio: copyCharRatio, output_format_ok: outputFormatOk };
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
    orchestratorSessionKey,
    scenario,
    taskRow,
    model,
    runTimeoutSeconds = 600,
    experimentId = "O0-pre-A",
    negativeControl = null,
  } = params;

  const agentCount = scenario.agent_count ?? 3;
  const probeAgents = new Set(scenario.ttft_probe_agents ?? [agentCount - 1]);
  const outputs = {};
  const records = [];
  const runStartedAt = Date.now();

  for (let agentIndex = 0; agentIndex < agentCount; agentIndex += 1) {
    const templateKey = `agent_${agentIndex}`;
    let template = taskRow.agent_tasks?.[templateKey];
    if (!template) {
      throw new Error(`Task ${taskRow.task_id} missing agent_tasks.${templateKey}`);
    }

    if (negativeControl === "NC-1" && agentIndex === 1) {
      template = template.replace(/\{\{agent_0_current\}\}/g, "");
    }

    const variables = {
      user_question: taskRow.user_question ?? "",
      task_body: taskRow.task_body ?? "",
      ...Object.fromEntries(
        Object.entries(outputs).map(([key, value]) => [key, value]),
      ),
    };

    const taskText = renderTemplateStrict(template, variables);
    const taskHash = sha256Short(taskText);

    const spawnStartedAt = Date.now();
    const invokePayload = await client.invokeTool(orchestratorSessionKey, "sessions_spawn", {
      task: taskText,
      mode: "run",
      context: "isolated",
      lightContext: true,
      cleanup: "keep",
      expectsCompletionMessage: false,
      ...(model ? { model } : {}),
      runTimeoutSeconds,
    });

    const spawnResult = extractToolJson(invokePayload);
    assertSpawnAccepted(spawnResult, agentIndex);

    const childSessionKey = spawnResult.childSessionKey;
    const childRunId = spawnResult.runId;

    const ttftDeadlineMs = runTimeoutSeconds * 1000;
    const ttftPromise = startTtftCollection(client, {
      sessionKey: childSessionKey,
      runId: childRunId,
      sinceMs: spawnStartedAt,
      untilMs: spawnStartedAt + ttftDeadlineMs + 15_000,
      timeoutMs: ttftDeadlineMs + 15_000,
    });

    if (childRunId) {
      await client.agentWait(childRunId, runTimeoutSeconds * 1000);
    } else {
      await client.waitForTask(childSessionKey, {
        timeoutMs: runTimeoutSeconds * 1000,
      });
    }

    const messages = await client.getSessionMessages(childSessionKey);
    const outputText = extractAssistantText(messages);

    const ttftInfo = await ttftPromise;
    const resolvedTtft =
      ttftInfo?.ttft_ms != null
        ? ttftInfo
        : await collectTtftForSession(childSessionKey, {
            sinceMs: spawnStartedAt,
            untilMs: Date.now(),
            wallClockMs: Date.now() - spawnStartedAt,
            runId: childRunId,
            client,
          });

    outputs[`agent_${agentIndex}_current`] = outputText;
    const outputFormat = analyzeOutputFormat(outputText);

    const record = {
      experiment_id: experimentId,
      task_id: taskRow.task_id,
      run_id: params.runId,
      agent_index: agentIndex,
      node_id: String(agentIndex),
      probe: probeAgents.has(agentIndex),
      child_session_key: childSessionKey,
      child_run_id: childRunId ?? null,
      task_hash: taskHash,
      upstream_hashes:
        agentIndex > 0
          ? {
              agent_0_current: outputs.agent_0_current
                ? sha256Short(outputs.agent_0_current)
                : null,
              ...(agentIndex > 1
                ? {
                    agent_1_current: outputs.agent_1_current
                      ? sha256Short(outputs.agent_1_current)
                      : null,
                  }
                : {}),
            }
          : {},
      task_includes_upstream:
        agentIndex === 0
          ? true
          : agentIndex === 1
            ? taskText.includes(outputs.agent_0_current ?? "__missing__")
            : scenario.topology === "chain"
              ? taskText.includes(outputs.agent_1_current ?? "__missing__")
              : taskText.includes(outputs.agent_0_current ?? "__missing__") &&
                taskText.includes(outputs.agent_1_current ?? "__missing__"),
      output_text: outputText,
      output_len: outputText.length,
      ...outputFormat,
      ttft_ms: resolvedTtft.ttft_ms,
      ttft_source: resolvedTtft.source,
      ttft_fallback: resolvedTtft.fallback,
      ttft_note: resolvedTtft.note ?? null,
      e2e_agent_ms: Date.now() - spawnStartedAt,
      timestamp: new Date().toISOString(),
    };

    records.push(record);
  }

  return {
    experiment_id: experimentId,
    task_id: taskRow.task_id,
    run_id: params.runId,
    agent_count: agentCount,
    e2e_run_ms: Date.now() - runStartedAt,
    records,
    outputs,
  };
}

export async function connectGateway(options) {
  const client = await GatewayClient.create(options);
  await client.connect();
  return client;
}
