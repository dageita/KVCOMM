import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import { resolvePristineOnlyPath } from "../lib/clawbench-chain.mjs";

test("resolvePristineOnlyPath ignores stale workspace cart.py for repo tasks", () => {
  const root = mkdtempSync(join(tmpdir(), "clawbench-pristine-"));
  const workspaceDir = join(root, "run-config-loader");
  mkdirSync(join(root, "bench-pristine", "t2_config_loader", "tests"), { recursive: true });
  writeFileSync(
    join(root, "bench-pristine", "t2_config_loader", "tests", "test_config_loader.py"),
    "def test_x():\n    assert True\n",
    "utf8",
  );
  mkdirSync(workspaceDir, { recursive: true });
  writeFileSync(join(workspaceDir, "cart.py"), "# stale cart\n", "utf8");

  const previousStateDir = process.env.OPENCLAW_STATE_DIR;
  process.env.OPENCLAW_STATE_DIR = root;
  try {
    const taskRow = {
      task_id: "t2-config-loader",
      clawbench_ref: {
        family: "repo",
        asset_packs: ["t2_config_loader"],
      },
    };

    assert.equal(resolvePristineOnlyPath(taskRow, "cart.py", workspaceDir), null);
    const testsSrc = resolvePristineOnlyPath(taskRow, "tests", workspaceDir);
    assert.ok(testsSrc);
    assert.notEqual(resolve(testsSrc), resolve(join(workspaceDir, "tests")));
  } finally {
    if (previousStateDir === undefined) {
      delete process.env.OPENCLAW_STATE_DIR;
    } else {
      process.env.OPENCLAW_STATE_DIR = previousStateDir;
    }
    rmSync(root, { recursive: true, force: true });
  }
});
