import { randomUUID } from "node:crypto";
import { WebSocket } from "ws";
import {
  formatSessionsSpawnBlockedHelp,
  formatTokenMismatchHelp,
  resolveGatewayToken,
} from "./openclaw-config.mjs";

const MIN_PROTOCOL = 3;
const MAX_PROTOCOL = 4;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function rawDataToString(data) {
  if (typeof data === "string") {
    return data;
  }
  if (Buffer.isBuffer(data)) {
    return data.toString("utf8");
  }
  if (data instanceof ArrayBuffer) {
    return Buffer.from(data).toString("utf8");
  }
  return String(data);
}

/**
 * Minimal OpenClaw Gateway WebSocket client for benchmark drivers.
 * Token auth (OPENCLAW_GATEWAY_TOKEN) on loopback is sufficient for local bench.
 */
function enrichToolInvokeError(toolName, err) {
  const message = String(err?.message ?? err);
  if (
    toolName === "sessions_spawn" &&
    (message.includes("Tool not available") || message.includes("not_found"))
  ) {
    return new Error(`${message}\n\n${formatSessionsSpawnBlockedHelp("~/.openclaw/openclaw.json")}`);
  }
  return err instanceof Error ? err : new Error(message);
}

export class GatewayClient {
  constructor(options = {}) {
    this.url =
      options.url ||
      process.env.OPENCLAW_GATEWAY_URL ||
      `ws://127.0.0.1:${process.env.OPENCLAW_GATEWAY_PORT || "18789"}`;
    this.token = options.token ?? "";
    this.tokenSource = options.tokenSource ?? "unset";
    this.requestTimeoutMs = options.requestTimeoutMs ?? 120_000;
    this.ws = null;
    this.pending = new Map();
    this.events = [];
    this.eventWaiters = [];
  }

  static async create(options = {}) {
    const resolved = await resolveGatewayToken(options.token);
    const client = new GatewayClient({
      ...options,
      token: resolved.token,
      tokenSource: resolved.source,
    });
    if (resolved.token) {
      console.log(`[gateway] auth token loaded from ${resolved.source}`);
    } else {
      console.warn(
        "[gateway] no auth token — set OPENCLAW_GATEWAY_TOKEN or ensure ~/.openclaw/openclaw.json has gateway.auth.token",
      );
    }
    return client;
  }

  async connect() {
    if (this.ws) {
      return;
    }
    const httpOrigin = this.url.replace(/^ws:\/\//, "http://").replace(/^wss:\/\//, "https://");
    await new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url, {
        handshakeTimeout: 30_000,
        headers: { Origin: httpOrigin },
      });
      this.ws = ws;
      ws.on("open", () => resolve());
      ws.on("error", (err) => reject(err));
      ws.on("message", (data) => this.#onMessage(rawDataToString(data)));
      ws.on("close", () => {
        for (const [, future] of this.pending) {
          future.reject(new Error("Gateway WebSocket closed"));
        }
        this.pending.clear();
      });
    });

    const challenge = await this.#waitForEvent("connect.challenge", 30_000);
    const challengeNonce =
      typeof challenge?.payload?.nonce === "string" ? challenge.payload.nonce.trim() : "";

    // ConnectParams schema: nonce belongs under `device`, not at root.
    // Token-only loopback auth does not require a device block (see ClawBench client).
    const connectParams = {
      minProtocol: MIN_PROTOCOL,
      maxProtocol: MAX_PROTOCOL,
      client: {
        id: "openclaw-control-ui",
        version: "kvcomm-bench-0.1",
        platform: "linux",
        mode: "ui",
      },
      role: "operator",
      scopes: [
        "operator.admin",
        "operator.read",
        "operator.write",
        "operator.approvals",
        "operator.pairing",
      ],
      caps: [],
      ...(this.token ? { auth: { token: this.token } } : {}),
    };

    if (challengeNonce && process.env.BENCH_GATEWAY_DEVICE === "1") {
      console.warn(
        "[gateway] BENCH_GATEWAY_DEVICE=1 set but device signing is not implemented; using token auth only",
      );
    }

    const hello = await this.request("connect", connectParams, { timeoutMs: 30_000 });
    if (hello?.type !== "hello-ok") {
      throw new Error(`Gateway connect failed: ${JSON.stringify(hello)}`);
    }
  }

  async close() {
    if (!this.ws) {
      return;
    }
    await new Promise((resolve) => {
      this.ws.once("close", resolve);
      this.ws.close();
    });
    this.ws = null;
  }

  async request(method, params = {}, options = {}) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("Gateway client is not connected");
    }
    const id = randomUUID();
    const frame = { type: "req", id, method, params };
    const timeoutMs = options.timeoutMs ?? this.requestTimeoutMs;

    const response = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`RPC ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });

      this.ws.send(JSON.stringify(frame));
    });

    if (!response.ok) {
      const err = response.error ?? {};
      const message = err.message ?? "unknown error";
      if (
        String(message).includes("token mismatch") ||
        String(err.code).includes("UNAUTHORIZED") ||
        String(message).includes("unauthorized")
      ) {
        throw new Error(`${message}\n\n${formatTokenMismatchHelp()}`);
      }
      throw new Error(`RPC ${method} failed: ${err.code ?? "?"} - ${message}`);
    }
    return response.payload;
  }

  async createSession({ agentId, model, label } = {}) {
    const payload = await this.request("sessions.create", {
      ...(agentId ? { agentId } : {}),
      ...(model ? { model } : {}),
      ...(label ? { label } : {}),
    });
    const key = payload.sessionKey || payload.key;
    if (!key) {
      throw new Error(`sessions.create returned no session key: ${JSON.stringify(payload)}`);
    }
    return key;
  }

  async invokeTool(sessionKey, name, args = {}) {
    try {
      const payload = await this.request("tools.invoke", {
        sessionKey,
        name,
        args,
        confirm: false,
      });
      return payload;
    } catch (err) {
      throw enrichToolInvokeError(name, err);
    }
  }

  async agentWait(runId, timeoutMs) {
    try {
      return await this.request(
        "agent.wait",
        { runId, timeoutMs: Math.max(1, timeoutMs) },
        { timeoutMs: timeoutMs + 15_000 },
      );
    } catch (err) {
      return { status: "error", error: String(err?.message ?? err) };
    }
  }

  async getSessionMessages(sessionKey) {
    try {
      const payload = await this.request("sessions.get", { key: sessionKey });
      return Array.isArray(payload.messages) ? payload.messages : [];
    } catch {
      return [];
    }
  }

  async waitForTask(sessionKey, { timeoutMs = 120_000, pollMs = 250 } = {}) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const payload = await this.request("tasks.list", {
        sessionKey,
        limit: 20,
      });
      const tasks = Array.isArray(payload.tasks) ? payload.tasks : [];
      const terminal = tasks.find((task) =>
        ["succeeded", "failed", "timed_out", "cancelled"].includes(task.status),
      );
      if (terminal) {
        return terminal;
      }
      await sleep(pollMs);
    }
    throw new Error(`Timeout waiting for terminal task on session ${sessionKey}`);
  }

  /**
   * Wait for the first assistant token on Gateway WS streams (agent/chat events).
   * OPENCLAW_DIAGNOSTICS_TIMELINE_PATH does not contain model.call.completed — this is the reliable path.
   */
  waitForFirstAssistantToken(runId, { startedAt, timeoutMs = 120_000, sessionKey } = {}) {
    if (!runId) {
      return Promise.resolve(null);
    }
    const deadline = startedAt + timeoutMs;
    const matches = (frame) => {
      if (frame?.type !== "event") {
        return false;
      }
      const payload = frame.payload;
      if (!payload || typeof payload !== "object") {
        return false;
      }
      if (payload.runId !== runId) {
        return false;
      }
      if (sessionKey && payload.sessionKey && payload.sessionKey !== sessionKey) {
        return false;
      }
      if (frame.event === "agent" && payload.stream === "assistant") {
        return hasAssistantStreamContent(payload.data);
      }
      if (frame.event === "chat") {
        if (payload.state === "delta" && hasAssistantStreamContent(payload.message?.content)) {
          return true;
        }
        if (payload.state === "delta" && typeof payload.delta === "string" && payload.delta.trim()) {
          return true;
        }
      }
      return false;
    };

    for (const frame of this.events) {
      if (matches(frame)) {
        return Promise.resolve({
          ttft_ms: Math.max(0, Date.now() - startedAt),
          observed_at: Date.now(),
          source: "gateway.ws",
          event: frame.event,
        });
      }
    }

    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        resolve(null);
      }, Math.max(0, deadline - Date.now()));
      this.eventWaiters.push({
        predicate: (frame) => {
          if (!matches(frame)) {
            return false;
          }
          clearTimeout(timer);
          resolve({
            ttft_ms: Math.max(0, Date.now() - startedAt),
            observed_at: Date.now(),
            source: "gateway.ws",
            event: frame.event,
          });
          return true;
        },
        resolve: () => {},
      });
    });
  }

  async diagnosticsStability({ type, limit = 200 } = {}) {
    try {
      return await this.request("diagnostics.stability", {
        ...(type ? { type } : {}),
        limit,
      });
    } catch {
      return null;
    }
  }

  #onMessage(raw) {
    let frame;
    try {
      frame = JSON.parse(raw);
    } catch {
      return;
    }

    if (frame.type === "res") {
      const pending = this.pending.get(frame.id);
      if (pending) {
        this.pending.delete(frame.id);
        pending.resolve(frame);
      }
      return;
    }

    if (frame.type === "event") {
      this.events.push(frame);
      for (const waiter of this.eventWaiters.splice(0)) {
        try {
          if (!waiter.predicate(frame)) {
            this.eventWaiters.push(waiter);
            continue;
          }
          waiter.resolve(frame);
        } catch {
          this.eventWaiters.push(waiter);
        }
      }
    }
  }

  #waitForEvent(eventName, timeoutMs) {
    const existing = this.events.find((event) => event.event === eventName);
    if (existing) {
      return Promise.resolve(existing);
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Timeout waiting for gateway event ${eventName}`));
      }, timeoutMs);
      this.eventWaiters.push({
        predicate: (frame) => frame.event === eventName,
        resolve: (frame) => {
          clearTimeout(timer);
          resolve(frame);
        },
      });
    });
  }
}

function hasAssistantStreamContent(data) {
  if (!data) {
    return false;
  }
  if (typeof data === "string" && data.trim()) {
    return true;
  }
  if (typeof data !== "object") {
    return false;
  }
  if (typeof data.delta === "string" && data.delta.trim()) {
    return true;
  }
  if (typeof data.text === "string" && data.text.trim()) {
    return true;
  }
  if (Array.isArray(data)) {
    return data.some((block) => hasAssistantStreamContent(block));
  }
  if (Array.isArray(data.content)) {
    return data.content.some(
      (block) =>
        block &&
        typeof block === "object" &&
        typeof block.text === "string" &&
        block.text.trim(),
    );
  }
  return false;
}

export function extractToolJson(payload) {
  if (!payload) {
    return null;
  }
  if (payload.result && typeof payload.result === "object") {
    return payload.result;
  }
  const output = payload.output ?? payload;
  if (typeof output === "string") {
    try {
      return JSON.parse(output);
    } catch {
      return { raw: output };
    }
  }
  if (output && typeof output === "object") {
    const content = output.content ?? output.details?.content;
    if (Array.isArray(content)) {
      const textBlock = content.find(
        (block) => block && typeof block === "object" && block.type === "text",
      );
      if (textBlock?.text) {
        try {
          return JSON.parse(textBlock.text);
        } catch {
          return { raw: textBlock.text };
        }
      }
    }
    if (typeof output.text === "string") {
      try {
        return JSON.parse(output.text);
      } catch {
        return { raw: output.text };
      }
    }
    return output;
  }
  return null;
}

export function extractAssistantText(messages) {
  const assistants = messages.filter((msg) => msg?.role === "assistant");
  const last = assistants.at(-1);
  if (!last) {
    return "";
  }
  if (typeof last.content === "string") {
    return last.content.trim();
  }
  if (Array.isArray(last.content)) {
    return last.content
      .map((block) => (typeof block?.text === "string" ? block.text : ""))
      .join("\n")
      .trim();
  }
  return "";
}
