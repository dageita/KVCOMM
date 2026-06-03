import { createReadStream } from "node:fs";
import { access, readFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

/**
 * Resolve diagnostics timeline / log paths (same env vars as OpenClaw gateway).
 * Note: timeline JSONL contains span/provider events only — NOT model.call.completed.
 */
export function resolveTimelinePath(env = process.env) {
  if (env.OPENCLAW_DIAGNOSTICS_TIMELINE_PATH?.trim()) {
    return env.OPENCLAW_DIAGNOSTICS_TIMELINE_PATH.trim();
  }
  const stateDir = env.OPENCLAW_STATE_DIR?.trim() || join(homedir(), ".openclaw");
  return join(stateDir, "diagnostics", "timeline.jsonl");
}

export function resolveDiagnosticsLogPath(env = process.env) {
  if (env.OPENCLAW_DIAGNOSTICS_LOG?.trim()) {
    return env.OPENCLAW_DIAGNOSTICS_LOG.trim();
  }
  const today = new Date().toISOString().slice(0, 10);
  return join(tmpdir(), "openclaw", `openclaw-${today}.log`);
}

async function fileExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function pickTtftFromStabilityEvent(event) {
  if (typeof event?.timeToFirstByteMs === "number" && Number.isFinite(event.timeToFirstByteMs)) {
    return event.timeToFirstByteMs;
  }
  return undefined;
}

function inTimeWindow(ts, sinceMs, untilMs) {
  if (!Number.isFinite(ts)) {
    return true;
  }
  return ts >= sinceMs - 5000 && ts <= untilMs + 5000;
}

async function collectFromDiagnosticsStability(client, { sinceMs, untilMs }) {
  const snapshot = await client.diagnosticsStability({
    type: "model.call.completed",
    limit: 1000,
  });
  const events = Array.isArray(snapshot?.events) ? snapshot.events : [];
  const hits = events.filter((event) => {
    const ttft = pickTtftFromStabilityEvent(event);
    if (ttft == null) {
      return false;
    }
    return inTimeWindow(event.ts, sinceMs, untilMs);
  });
  if (hits.length === 0) {
    return null;
  }
  const last = hits.at(-1);
  return {
    ttft_ms: pickTtftFromStabilityEvent(last),
    source: "diagnostics.stability",
    path: "gateway:diagnostics.stability",
    event_type: last.type,
    fallback: false,
    note: "matched by time window only (stability records omit sessionKey)",
  };
}

async function collectFromGatewayWs(client, { runId, sessionKey, startedAt, timeoutMs }) {
  if (!client?.waitForFirstAssistantToken || !runId) {
    return null;
  }
  const hit = await client.waitForFirstAssistantToken(runId, {
    startedAt,
    timeoutMs,
    sessionKey,
  });
  if (!hit?.ttft_ms && hit?.ttft_ms !== 0) {
    return null;
  }
  return {
    ttft_ms: hit.ttft_ms,
    source: hit.source ?? "gateway.ws",
    path: `gateway.ws:${hit.event ?? "agent"}`,
    event_type: hit.event ?? "agent",
    fallback: false,
  };
}

/**
 * Collect TTFT for a subagent run.
 *
 * Priority:
 *  1. Gateway WS agent/chat stream (first assistant delta) — recommended
 *  2. diagnostics.stability model.call.completed (time window; often empty — trusted events excluded)
 *  3. wall-clock fallback (equals e2e_agent_ms; not true TTFT)
 */
export async function collectTtftForSession(sessionKey, options = {}) {
  const {
    sinceMs = 0,
    untilMs = Date.now() + 60_000,
    wallClockMs,
    runId = null,
    client = null,
    timeoutMs = Math.max(0, untilMs - sinceMs),
  } = options;

  if (client && runId) {
    const wsHit = await collectFromGatewayWs(client, {
      runId,
      sessionKey,
      startedAt: sinceMs,
      timeoutMs,
    });
    if (wsHit) {
      return wsHit;
    }
  }

  if (client) {
    const stabilityHit = await collectFromDiagnosticsStability(client, { sinceMs, untilMs });
    if (stabilityHit) {
      return stabilityHit;
    }
  }

  return {
    ttft_ms: wallClockMs ?? null,
    source: "wall_clock_fallback",
    path: null,
    event_type: null,
    fallback: true,
    note:
      "No WS stream token observed. OPENCLAW_DIAGNOSTICS_TIMELINE_PATH records spans only, not TTFT.",
  };
}

/**
 * Start TTFT collection concurrently with agent.wait — call before waiting for run completion.
 */
export function startTtftCollection(client, params) {
  const { sessionKey, runId, sinceMs, untilMs, timeoutMs } = params;
  return collectTtftForSession(sessionKey, {
    client,
    runId,
    sinceMs,
    untilMs,
    timeoutMs,
    wallClockMs: null,
  });
}
