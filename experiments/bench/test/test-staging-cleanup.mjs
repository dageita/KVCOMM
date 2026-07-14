import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

import {
  collectTaskCodingArtifacts,
  purgeStaleCodingArtifacts,
  resolveDefaultAgentWorkspace,
} from "../lib/clawbench-chain.mjs";

const CONFIG_LOADER_TASK = {
  task_id: "t2-config-loader",
  clawbench_ref: {
    family: "repo",
    asset_packs: ["t2_config_loader"],
  },
};

const BUGFIX_TASK = {
  task_id: "t1-bugfix-discount",
  clawbench_ref: {
    family: "coding",
    asset_packs: ["t1_bugfix_discount"],
  },
};

test("collectTaskCodingArtifacts reads asset pack basenames", async () => {
  const { allowedRoot, allowedTests } = await collectTaskCodingArtifacts(CONFIG_LOADER_TASK);
  assert.equal(allowedRoot.has("config_loader.py"), true);
  assert.equal(allowedRoot.has("app_config.py"), true);
  assert.equal(allowedRoot.has("pricing.py"), false);
  assert.equal(allowedTests.has("test_config_loader.py"), true);
  assert.equal(allowedTests.has("test_pricing.py"), false);
});

test("purgeStaleCodingArtifacts removes cross-task pollution for config-loader", async () => {
  const root = await mkdtemp(join(tmpdir(), "clawbench-purge-"));
  try {
    await mkdir(join(root, "tests"), { recursive: true });
    await writeFile(join(root, "config_loader.py"), "# task fixture\n", "utf8");
    await writeFile(join(root, "app_config.py"), "DEFAULTS = {}\n", "utf8");
    await writeFile(join(root, "tests", "test_config_loader.py"), "def test_x(): pass\n", "utf8");
    await writeFile(join(root, "test_text_normalization.py"), "from text_normalization_module import x\n", "utf8");
    await writeFile(join(root, "normalizer.py"), "# stale\n", "utf8");
    await writeFile(join(root, "pricing.py"), "# stale\n", "utf8");
    await writeFile(join(root, "cart.py"), "# stale\n", "utf8");
    await writeFile(join(root, "tests", "test_pricing.py"), "def test_x(): pass\n", "utf8");

    const { removed } = await purgeStaleCodingArtifacts(root, CONFIG_LOADER_TASK);
    assert.ok(removed.includes("test_text_normalization.py"));
    assert.ok(removed.includes("normalizer.py"));
    assert.ok(removed.includes("pricing.py"));
    assert.ok(removed.includes("cart.py"));
    assert.ok(removed.includes("tests/test_pricing.py"));

    assert.equal(existsSync(join(root, "config_loader.py")), true);
    assert.equal(existsSync(join(root, "tests", "test_config_loader.py")), true);
    assert.equal(existsSync(join(root, "test_text_normalization.py")), false);
    assert.equal(existsSync(join(root, "pricing.py")), false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("purgeStaleCodingArtifacts keeps agent-authored test_normalizer for mutable-tests task", async () => {
  const root = await mkdtemp(join(tmpdir(), "clawbench-purge-normalizer-"));
  try {
    await mkdir(join(root, "tests"), { recursive: true });
    await writeFile(join(root, "normalizer.py"), "# fixture\n", "utf8");
    await writeFile(
      join(root, "tests", "test_normalizer.py"),
      "from normalizer import normalize_title\ndef test_x(): pass\n",
      "utf8",
    );
    await writeFile(join(root, "tests", "test_pricing.py"), "def test_stale(): pass\n", "utf8");
    await writeFile(join(root, "pricing.py"), "# stale\n", "utf8");

    const task = {
      task_id: "t2-add-tests-normalizer",
      clawbench_ref: {
        family: "coding",
        asset_packs: ["t2_add_tests_normalizer"],
      },
    };
    const { removed } = await purgeStaleCodingArtifacts(root, task);
    assert.ok(removed.includes("pricing.py"));
    assert.ok(removed.includes("tests/test_pricing.py"));
    assert.equal(removed.includes("tests/test_normalizer.py"), false);
    assert.equal(existsSync(join(root, "tests", "test_normalizer.py")), true);
    assert.equal(existsSync(join(root, "normalizer.py")), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("purgeStaleCodingArtifacts keeps bugfix task fixtures", async () => {
  const root = await mkdtemp(join(tmpdir(), "clawbench-purge-bugfix-"));
  try {
    await mkdir(join(root, "tests"), { recursive: true });
    await writeFile(join(root, "pricing.py"), "# fixture\n", "utf8");
    await writeFile(join(root, "cart.py"), "# fixture\n", "utf8");
    await writeFile(join(root, "tests", "test_pricing.py"), "def test_x(): pass\n", "utf8");
    await writeFile(join(root, "config_loader.py"), "# stale\n", "utf8");

    const { removed } = await purgeStaleCodingArtifacts(root, BUGFIX_TASK);
    assert.ok(removed.includes("config_loader.py"));
    assert.equal(existsSync(join(root, "pricing.py")), true);
    assert.equal(existsSync(join(root, "cart.py")), true);
    assert.equal(existsSync(join(root, "tests", "test_pricing.py")), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("resolveDefaultAgentWorkspace follows OPENCLAW_STATE_DIR", async () => {
  const root = await mkdtemp(join(tmpdir(), "clawbench-state-"));
  const previous = process.env.OPENCLAW_STATE_DIR;
  process.env.OPENCLAW_STATE_DIR = root;
  try {
    assert.equal(resolveDefaultAgentWorkspace(), join(root, "workspace"));
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCLAW_STATE_DIR;
    } else {
      process.env.OPENCLAW_STATE_DIR = previous;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("bare pytest -q collects only active task tests after purge", async () => {
  const root = await mkdtemp(join(tmpdir(), "clawbench-pytest-"));
  try {
    const tasksPublic = join(process.cwd(), "../../../clawbench/tasks-public/assets/t2_config_loader");
    const workspace = join(root, "run-config-loader");
    await mkdir(workspace, { recursive: true });
    for (const name of ["config_loader.py", "app_config.py"]) {
      await writeFile(join(workspace, name), await readFile(join(tasksPublic, name), "utf8"), "utf8");
    }
    await mkdir(join(workspace, "tests"), { recursive: true });
    await writeFile(
      join(workspace, "tests", "test_config_loader.py"),
      await readFile(join(tasksPublic, "tests", "test_config_loader.py"), "utf8"),
      "utf8",
    );
    await writeFile(join(workspace, "test_text_normalization.py"), "import pytest\n", "utf8");

    await purgeStaleCodingArtifacts(workspace, CONFIG_LOADER_TASK);

    const result = spawnSync("pytest", ["-q", "--collect-only"], {
      cwd: workspace,
      env: {
        ...process.env,
        PYTHONPATH: [workspace, process.env.PYTHONPATH].filter(Boolean).join(":"),
      },
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout, /test_config_loader\.py/);
    assert.doesNotMatch(result.stdout + result.stderr, /test_text_normalization/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
