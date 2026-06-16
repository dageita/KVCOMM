/**
 * Build parameterized multi-agent scenarios and extend Chain task templates for N agents.
 */

const DEFAULT_UPSTREAM_LABEL = "Copy Machine";

export function buildChainAgentTemplate(agentIndex, options = {}) {
  const {
    roleSlot = "{{copy_role}}",
    taskIntro = "The task is: {{task_body}}\n",
    upstreamLabel = DEFAULT_UPSTREAM_LABEL,
    agentRoles = [],
  } = options;

  if (agentIndex === 0) {
    return `${roleSlot}\n\n${taskIntro}`;
  }

  const prevIndex = agentIndex - 1;
  const prevRole = agentRoles[prevIndex] ?? upstreamLabel;
  return (
    `${roleSlot}\n\n${taskIntro}` +
    "At the same time, the outputs of other agents are as follows:\n\n" +
    `Agent ${prevIndex}, role is ${prevRole}, output is:\n\n {{agent_${prevIndex}_current}}\n\n`
  );
}

/**
 * @param {object} base - optional base scenario JSON
 * @param {number} agentCount
 * @param {string} topology - currently only "chain"
 */
export function buildScenario(base = {}, agentCount = 3, topology = "chain") {
  if (agentCount < 1) {
    throw new Error(`agent_count must be >= 1, got ${agentCount}`);
  }
  if (topology !== "chain") {
    throw new Error(`topology "${topology}" not implemented yet (only chain)`);
  }

  const baseCount = base.agent_count ?? agentCount;
  const countChanged = agentCount !== baseCount;
  const agent_ids = Array.from({ length: agentCount }, (_, i) => `agent_${i}`);
  return {
    id: countChanged ? `${agentCount}agent-${topology}-v1` : (base.id ?? `${agentCount}agent-${topology}-v1`),
    topology,
    agent_count: agentCount,
    agent_ids,
    ttft_probe_agents: countChanged
      ? [agentCount - 1]
      : (base.ttft_probe_agents ?? [agentCount - 1]),
    generation: base.generation ?? { max_tokens: 512, temperature: 0 },
    notes: base.notes ?? `Generated ${agentCount}-agent ${topology} scenario`,
  };
}

function hasCompleteAgentTemplates(taskRow, agentCount) {
  if (!taskRow.agent_tasks || typeof taskRow.agent_tasks !== "object") {
    return false;
  }
  for (let i = 0; i < agentCount; i += 1) {
    const key = `agent_${i}`;
    if (!taskRow.agent_tasks[key] || typeof taskRow.agent_tasks[key] !== "string") {
      return false;
    }
  }
  return true;
}

/**
 * Ensure task row has agent_0..agent_{N-1} templates (Chain spatial mask).
 * Does not overwrite existing complete templates.
 */
export function extendTaskAgentTemplates(taskRow, agentCount, topology = "chain", options = {}) {
  if (topology !== "chain") {
    return taskRow;
  }

  if (hasCompleteAgentTemplates(taskRow, agentCount)) {
    return taskRow;
  }

  const agentRoles = taskRow.agent_roles ?? [];
  const isClawbench = Boolean(taskRow.clawbench_ref);
  const roleSlot = isClawbench ? "{{role_prompt}}" : "{{copy_role}}";
  const taskIntro = isClawbench
    ? "User request:\n{{task_body}}\n"
    : "The task is: {{task_body}}\n";

  const agent_tasks = { ...(taskRow.agent_tasks ?? {}) };
  for (let i = 0; i < agentCount; i += 1) {
    const key = `agent_${i}`;
    if (!agent_tasks[key]) {
      agent_tasks[key] = buildChainAgentTemplate(i, {
        roleSlot,
        taskIntro,
        upstreamLabel: agentRoles[i - 1] ?? DEFAULT_UPSTREAM_LABEL,
        agentRoles,
      });
    }
  }
  return { ...taskRow, agent_tasks };
}
