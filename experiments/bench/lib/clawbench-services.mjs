import { createServer } from "node:net";
import { spawn } from "node:child_process";
import { join } from "node:path";

async function pickFreePort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.listen(0, "127.0.0.1", () => {
      const port = server.address()?.port;
      server.close((err) => (err ? reject(err) : resolve(port)));
    });
    server.on("error", reject);
  });
}

async function waitForHealth(baseUrl, { readyPath = "/health", timeoutMs = 20_000 } = {}) {
  const url = `${baseUrl.replace(/\/$/, "")}${readyPath.startsWith("/") ? readyPath : `/${readyPath}`}`;
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // retry until timeout
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Background service not ready at ${url} within ${timeoutMs}ms`);
}

/**
 * Start task background services (e.g. form_app for browser bench tasks).
 * Returns runtime template values like form_app_port.
 */
export async function startTaskBackgroundServices(taskRow, workspaceDir) {
  const specs = taskRow?.clawbench_ref?.background_services ?? [];
  if (!specs.length) {
    return { runtimeValues: {}, stop: async () => {} };
  }

  const managed = [];
  const runtimeValues = {};

  for (const spec of specs) {
    const name = String(spec.name || "service");
    const port = Number(spec.port) > 0 ? Number(spec.port) : await pickFreePort();
    runtimeValues[`${name}_port`] = port;
    runtimeValues[`${name}_url`] = `http://127.0.0.1:${port}`;

    const command = String(spec.command || "").trim();
    if (!command) {
      throw new Error(`Background service ${name} missing command`);
    }
    const [bin, ...args] = command.split(/\s+/);
    const cwd = workspaceDir;
    const env = {
      ...process.env,
      ...(spec.port_env ? { [spec.port_env]: String(port) } : { PORT: String(port) }),
      ...Object.fromEntries(
        Object.entries(spec.env || {}).map(([key, value]) => [key, String(value)]),
      ),
    };

    const proc = spawn(bin, args, {
      cwd,
      env,
      stdio: "ignore",
    });
    managed.push(proc);

    const readyPath = spec.ready_path ?? "/health";
    await waitForHealth(`http://127.0.0.1:${port}`, {
      readyPath,
      timeoutMs: (spec.startup_timeout_seconds ?? 20) * 1000,
    });
  }

  return {
    runtimeValues,
    async stop() {
      for (const proc of managed) {
        proc.kill("SIGTERM");
      }
    },
  };
}
