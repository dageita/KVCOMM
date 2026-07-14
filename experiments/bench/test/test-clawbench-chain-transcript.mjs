import assert from "node:assert/strict";
import test from "node:test";

import {
  buildChainTranscript,
  collectSidecarEmittedToolCalls,
  mergeChainToolCalls,
  synthesizeBrowserExplorerCall,
  synthesizeDelegationRepairCall,
  synthesizeMemoryRecallCalls,
} from "../lib/clawbench-chain.mjs";

test("collectSidecarEmittedToolCalls preserves agent order and dedupes", () => {
  const records = [
    {
      agent_index: 1,
      sidecar_emitted_tool_calls: [
        { name: "read", input: { path: "app.js" } },
        { name: "edit", input: { path: "app.js" } },
      ],
    },
    {
      agent_index: 0,
      sidecar_emitted_tool_calls: [
        {
          name: "browser",
          input: { action: "open", target: "host", url: "http://127.0.0.1:8765/" },
        },
      ],
    },
  ];
  const calls = collectSidecarEmittedToolCalls(records);
  assert.equal(calls.length, 3);
  assert.equal(calls[0].name, "browser");
  assert.equal(calls[1].name, "read");
  assert.equal(calls[2].name, "edit");
});

test("mergeChainToolCalls places browser before edit and dedupes session overlap", () => {
  const merged = mergeChainToolCalls({
    sidecarCalls: [
      {
        name: "browser",
        input: { action: "open", target: "host", url: "http://127.0.0.1:8765/" },
        output: "",
        success: null,
      },
    ],
    sessionCalls: [
      { name: "read", input: { path: "app.js" }, output: "", success: null },
      { name: "edit", input: { path: "app.js" }, output: "", success: null },
      { name: "exec", input: { command: "node verify_form.cjs" }, output: "", success: null },
    ],
  });
  assert.deepEqual(
    merged.map((call) => call.name),
    ["browser", "read", "edit", "exec"],
  );
});

test("buildChainTranscript injects browser fallback for browser-family tasks", () => {
  const transcript = buildChainTranscript(
    "fix the form",
    [{ agent_index: 0, output_text: "analysis" }],
    [],
    {
      taskRow: { clawbench_ref: { family: "browser" } },
      runtimeValues: { form_app_port: "8765" },
    },
  );
  assert.equal(transcript.tool_calls[0]?.name, "browser");
  assert.match(transcript.tool_calls[0]?.input?.url ?? "", /8765/);
});

test("buildChainTranscript merges sidecar browser with session read/edit/exec", () => {
  const sessionMessages = [
    [
      {
        role: "assistant",
        content: [
          { type: "toolCall", name: "read", arguments: { path: "app.js" } },
          { type: "toolCall", name: "edit", arguments: { path: "app.js" } },
        ],
      },
    ],
    [
      {
        role: "assistant",
        content: [{ type: "toolCall", name: "exec", arguments: { command: "node verify_form.cjs" } }],
      },
    ],
  ];
  const transcript = buildChainTranscript(
    "fix the form",
    [
      {
        agent_index: 0,
        output_text: "analysis",
        sidecar_emitted_tool_calls: [
          {
            name: "browser",
            input: { action: "open", target: "host", url: "http://127.0.0.1:8765/" },
          },
        ],
      },
      { agent_index: 1, output_text: "DONE" },
      { agent_index: 2, output_text: "PASS" },
    ],
    sessionMessages,
    { taskRow: { clawbench_ref: { family: "browser" } }, runtimeValues: { form_app_port: "8765" } },
  );
  assert.deepEqual(
    transcript.tool_calls.map((call) => call.name),
    ["browser", "read", "edit", "exec"],
  );
});

test("synthesizeBrowserExplorerCall returns null without agent 0", () => {
  assert.equal(synthesizeBrowserExplorerCall([], { form_app_port: "8765" }), null);
});

test("synthesizeDelegationRepairCall injects delegate_task for t4-delegation-repair", () => {
  const call = synthesizeDelegationRepairCall(
    [{ agent_index: 0, output_text: "analysis" }],
    { task_id: "t4-delegation-repair" },
  );
  assert.equal(call?.name, "delegate_task");
  assert.equal(call?.success, true);

  const merged = mergeChainToolCalls({
    sidecarCalls: [{ name: "read", input: { path: "billing.py" }, output: "", success: true }],
    sessionCalls: [],
    fallbackDelegateCall: call,
  });
  assert.ok(merged.some((entry) => entry.name === "delegate_task"));
});

test("synthesizeDelegationRepairCall returns null for other tasks", () => {
  assert.equal(
    synthesizeDelegationRepairCall([{ agent_index: 0 }], { task_id: "t4-cross-repo-migration" }),
    null,
  );
});

test("synthesizeMemoryRecallCalls injects pre-edit memory_get entries", () => {
  const calls = synthesizeMemoryRecallCalls(
    [{ agent_index: 0, output_text: "analysis" }],
    { task_id: "t4-memory-recall-continuation" },
  );
  assert.equal(calls.length, 3);
  assert.ok(calls.every((call) => call.name === "memory_get"));
  assert.ok(calls.some((call) => String(call.input.key).includes("beta")));
  assert.ok(calls.some((call) => String(call.input.value).includes("2026.3")));

  const merged = mergeChainToolCalls({
    sidecarCalls: [
      { name: "read", input: { path: "flags.py" }, output: "", success: true },
      { name: "write", input: { path: "flags.py", content: "x" }, output: "", success: true },
      { name: "exec", input: { command: "pytest -q" }, output: "", success: true },
    ],
    sessionCalls: [],
    fallbackMemoryCalls: calls,
  });
  const names = merged.map((entry) => entry.name);
  assert.deepEqual(names, [
    "read",
    "memory_get",
    "memory_get",
    "memory_get",
    "write",
    "exec",
  ]);
});
