import assert from "node:assert/strict";
import test from "node:test";

import { isCodingLikeTask } from "../lib/clawbench-chain.mjs";

test("isCodingLikeTask includes repo and coding families", () => {
  assert.equal(isCodingLikeTask({ clawbench_ref: { family: "coding" } }), true);
  assert.equal(isCodingLikeTask({ clawbench_ref: { family: "repo" } }), true);
  assert.equal(isCodingLikeTask({ task_id: "t2-config-loader", clawbench_ref: { family: "repo" } }), true);
  assert.equal(isCodingLikeTask({ clawbench_ref: { family: "browser" } }), false);
  assert.equal(isCodingLikeTask({ clawbench_ref: { family: "tools" } }), false);
});
