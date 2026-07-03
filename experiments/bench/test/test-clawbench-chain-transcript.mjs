import assert from "node:assert/strict";
import test from "node:test";

import {
  buildChainTranscript,
  collectSidecarEmittedToolCalls,
  mergeChainToolCalls,
  synthesizeBrowserExplorerCall,
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
