# OpenClaw Subagent + TTFT Bench Driver

> **主入口已迁至 [`../../openclaw/README.md`](../../openclaw/README.md)**（KVCOMM OpenClaw 集成模块）。
> 推荐使用：`cd ../../openclaw && node cli.mjs bench run ...`

基于 §9.2a.0d 的最小 harness：通过 Gateway `tools.invoke(sessions_spawn)` 栈驱动 N-agent Chain，并采集 per-agent TTFT。

## 目录

```
bench/
├── drivers/run-o0-pre-chain.mjs   # 主入口
├── lib/
│   ├── gateway-client.mjs         # Gateway WebSocket RPC
│   ├── spawn-stack.mjs            # Chain stack_spawn 编排
│   ├── ttft-collector.mjs         # diagnostics.timeline → ttft_ms
│   ├── template.mjs               # {{placeholder}} 渲染
│   └── load-jsonl.mjs
├── datasets/tier0_copy.jsonl
├── scenarios/3agent-chain.json
└── prompts/copy_machine.role.txt
```

## 前置条件

1. **OpenClaw Gateway** 已启动。TTFT 通过 **Gateway WebSocket `agent`/`chat` 流事件**采集，不依赖 timeline 文件：

```bash
OPENCLAW_DIAGNOSTICS=timeline \
OPENCLAW_DIAGNOSTICS_TIMELINE_PATH=/tmp/openclaw-kvcomm-timeline.jsonl \
openclaw gateway run
```

`OPENCLAW_DIAGNOSTICS_TIMELINE_PATH` 仅写入 span/provider 类事件，**不含** `model.call.completed` / `timeToFirstByteMs`。若结果里 `ttft_source=wall_clock_fallback`，说明 WS 流未观测到首 token（常见于 provider 非流式返回）。

2. **认证**：Gateway token 必须与 **正在运行的 Gateway 进程** 读取的 `openclaw.json` 里 `gateway.auth.token` 完全一致。

```bash
# 方式 A（推荐）：不手动 export，driver 自动读 ~/.openclaw/openclaw.json
unset OPENCLAW_GATEWAY_TOKEN
npm run run -- --runs 1 --task-id micro-001 --model custom-10-121-129-19-30001/MiniMax-M2.7

# 方式 B：显式 export（勿使用占位符 sk-local，须与配置文件一致）
grep -A2 '"auth"' ~/.openclaw/openclaw.json
export OPENCLAW_GATEWAY_TOKEN="<gateway.auth.token 的值>"
```

若 Gateway 日志出现 `reason=token_mismatch`，说明环境变量里的 token 与 Gateway 配置不一致。

**启动 Gateway 时** 请勿设置错误的 `OPENCLAW_GATEWAY_TOKEN=sk-local`；若该变量与 `openclaw.json` 的 `gateway.auth.token` 不一致，会导致 **subagent spawn 内部连接失败**（即使 bench 客户端已连上）。

```bash
unset OPENCLAW_GATEWAY_TOKEN
OPENCLAW_DIAGNOSTICS=timeline \
OPENCLAW_DIAGNOSTICS_TIMELINE_PATH=/tmp/openclaw-kvcomm-timeline.jsonl \
openclaw gateway run
```

3. **Agent 工具 profile**：orchestrator 需要 `sessions_spawn`。`minimal` profile **不含**该工具，须加：

```json
"tools": {
  "profile": "minimal",
  "alsoAllow": ["sessions_spawn"]
}
```

或使用 `"profile": "coding"`（prompt 更大，需 vLLM ctx ≥8192）。

4. **Gateway tools.invoke 白名单**：OpenClaw 默认禁止通过 `tools.invoke` 调用 `sessions_spawn`（HTTP 安全 deny list）。本地 bench 须在 `openclaw.json` 中显式放行并 **重启 Gateway**：

```json
"gateway": {
  "tools": {
    "allow": ["sessions_spawn"]
  }
}
```

仅在本机 loopback bench 环境启用；勿对公网 Gateway 开放。

5. **模型**：`--model` 必须是 OpenClaw **已配置的 provider 引用**，不是本地目录路径。

```bash
# 正确（示例，以 openclaw.json agents.defaults.model.primary 为准）
--model custom-10-121-129-19-30001/MiniMax-M2.7

# 错误 — 本地 HF 权重路径 OpenClaw 无法直接解析
--model vllm/Qwen3-32B
```

若使用本地 Qwen3-32B，需先在 OpenClaw 中配置 vLLM/OpenAI-compat provider，再传 `vllm/Qwen3-32B`。

### Gateway 配置（bench 必需）

1. **`sessions_spawn` 白名单**（否则 bench 启动即失败）：

```json
"gateway": {
  "tools": {
    "allow": ["sessions_spawn"]
  }
}
```

修改后 **必须重启 Gateway**。

2. **模型 allowlist**：在 `agents.defaults.models` 注册 vLLM 模型，例如：

```json
"vllm/Qwen3-32B": {
  "params": { "extra_body": { "tool_choice": "none" } }
}
```

可合并片段：[`config/openclaw.models.snippet.json`](config/openclaw.models.snippet.json)（写入 `~/.openclaw/openclaw.json` 的 `agents.defaults.models`）。**修改后须冷重启 Gateway**。

3. **Orchestrator 工具（main agent only）**：bench 通过 `tools.invoke(sessions_spawn)` 驱动，**不要**在全局 `tools.allow` 写 `sessions_spawn`（subagent 会继承并报错）。推荐：

```json
"tools": {
  "profile": "minimal",
  "subagents": {
    "tools": { "allow": ["session_status"] }
  }
},
"agents": {
  "list": [{
    "id": "main",
    "tools": {
      "profile": "minimal",
      "alsoAllow": ["sessions_spawn"]
    }
  }]
},
"gateway": {
  "tools": { "allow": ["sessions_spawn"] }
}
```

4. **Context overflow 防护**：
   - Driver：每个 spawn 使用**全新 orchestrator session**（见 `spawn-stack.mjs`）
   - 长期 bench 前：`KVCOMM/openclaw/scripts/clean-bench-sessions.sh` 或 `bench run --clean-sessions`（建议 Gateway 已停止）
   - vLLM `--max-model-len 32768` + OpenClaw `contextWindow: 32768`

5. **Subagent 工具对齐（L1 限制）**：OpenClaw leaf subagent 若通过 `deny` 禁用全部 action 工具，会与 `gateway.tools.allow` 传入的 inherited allowlist 冲突，报 `No callable tools remain`。当前可用配置见上节 `subagents.tools.allow`。

修改 `openclaw.json` 后须 **冷重启 Gateway**（`openclaw gateway stop` → `openclaw gateway run`），热加载对 subagent 策略不可靠。

5. **vLLM 服务端要求**（常见报错）：

| 报错 | 原因 | 处理 |
|------|------|------|
| `sessions_spawn is blocked` | 未配置 `gateway.tools.allow` | 见上，重启 Gateway |
| `Tool not available: sessions_spawn` | main agent 用了 `allow` 而非 `alsoAllow` | `agents.list[main].tools.alsoAllow: ["sessions_spawn"]`（minimal profile 不含 spawn） |
| `model not allowed` | 未在 `agents.defaults.models` 注册 | 添加 `vllm/<model-id>` |
| `tool choice requires --enable-auto-tool-choice` | vLLM 未开 tool calling | bench 用 `tool_choice: none`（已配） |
| `No callable tools remain` | 全局 `tools.allow: [sessions_spawn]` 被 subagent 继承 | 仅 main agent + gateway.tools.allow 放行；subagent 用 `session_status` |
| Context overflow on `agent:main:main` | orchestrator session 历史膨胀 | 每 spawn 新 session；bench 前 `--clean-sessions` |

4. **Gateway 启动**（注意 Node 版本）：OpenClaw 需 Node ≥22.19；若 `node -v` 为 20.x，用 `/usr/bin/node`：

```bash
export PATH="/usr/bin:/bin:$PATH"
unset OPENCLAW_GATEWAY_TOKEN
export VLLM_API_KEY=vllm-local
openclaw gateway run
```

### vLLM 上下文长度（Context overflow 必读）

Gateway 日志若出现：

```
estimatedPromptTokens=5816 promptBudgetBeforeReserve=2048 overflowTokens=3768
Context overflow: prompt too large for the model (precheck)
```

说明 **OpenClaw 框架 system prompt（~5.8k tokens）已超过 vLLM 的 max_model_len**。这与 bench task 文本大小无关。

**vLLM 必须 `--max-model-len 16384`**（8192 不够，`coding` profile 框架 prompt ~5816 tokens）：

```bash
vllm serve /models/Qwen3-32B \
  --served-model-name Qwen3-32B \
  --dtype float16 \
  --max-model-len 16384 \
  --port 8001
```

OpenClaw 侧：`tools.profile: coding`，`contextWindow: 16384`，`compaction.reserveTokensFloor: 0`。

验证：`curl -s http://<vllm-host>:8001/v1/models` 中 `max_model_len` ≥ 16384。

## 与 KVCOMM `benchmark_TTFT.py` 对齐（不改 KVCOMM 代码）

OpenClaw 侧通过 `datasets/tier0_copy.jsonl` + `fixtures/kvcomm_tasks_seed42.json` 复现 benchmark 默认 workload：

| 维度 | KVCOMM（保持默认） | OpenClaw bench（已对齐） |
|------|-------------------|-------------------------|
| Task | `SEED=42` 下每次 sample 1000 个 `Δ`/`Ω`（空格分隔） | `kvcomm_task` 读同一 fixture；`advance_per_run: true` 时 run `i` 用 index `base+i` |
| 拓扑 | `Chain`，agent `i` 只见 predecessor `i-1` | agent_2 模板只注入 `agent_1_current`（非 agent_0+1） |
| Upstream 句式 | `Agent {id}, role is Copy Machine, output is:\n\n {out}` | 同上，见 `agent_1` / `agent_2` 模板 |
| User task 行 | `The task is: {task}\n` | `The task is: {{task_body}}\n` |
| Prefix / 输出指令 | `IN_LENGTH` / `OUT_LENGTH`（system） | `COPY_PREFIX_REPEATS` / `COPY_OUT_LENGTH`（拼进 spawn task） |
| 生成长度 | `DEFAULT_MAX_TOKENS` / `OUT_LENGTH` = 512 | 在 `openclaw.json` 模型上设 `max_tokens: 512`；`sessions_spawn` 无此参数 |

**推荐命令（与 `chain_3_512_512_default` 同预算）：**

```bash
cd KVCOMM/experiments/bench
export COPY_PREFIX_REPEATS=512 COPY_OUT_LENGTH=512
npm run dry-run
npm run run -- --runs 30 --task-id micro-001 --model vllm/Qwen3-32B \
  --output chain_3_512_openclaw_aligned
```

生成 `results/chain_3_512_openclaw_aligned.jsonl` 与 `.summary.json`（或 `BENCH_OUTPUT=同名`）。

- `micro-001` + `--runs 30`：task index 0..29，对应 KVCOMM `--samples 30` 的前 30 条随机 task（同 SEED=42）。
- 固定某次 KVCOMM run 的 task：在 jsonl 行上设 `"task_body": "<从 log REQUEST REUSE 的 task 字段粘贴>"`，并去掉 `kvcomm_task`。

**仍无法消除：** copy 约束在 KVCOMM 为 **system**，OpenClaw 在 **user task**；框架 system ~5.8k tokens；TTFT 口径不同。

## 快速开始

```bash
cd KVCOMM/experiments/bench
npm install

# 仅验证 dataset/scenario 渲染（无需 Gateway）
npm run dry-run

# 跑 1 条 task（与 KVCOMM sample 0 同 task 文本）
COPY_PREFIX_REPEATS=512 COPY_OUT_LENGTH=512 \
  npm run run -- --runs 1 --task-id micro-001 --model vllm/your-model

# 跑 30 次（对齐 KVCOMM --samples 30 的前 30 个 task）
COPY_PREFIX_REPEATS=512 COPY_OUT_LENGTH=512 \
  npm run run -- --runs 30 --task-id micro-001 --model vllm/your-model
```

## 工作原理

1. 创建 orchestrator session（`sessions.create`）
2. 对每个 agent 顺序调用 `tools.invoke` → `sessions_spawn`（`mode: run`, `context: isolated`）
3. 等待 child run 完成（`agent.wait` / `tasks.list`）
4. 从 child session 读取 assistant 输出 → 注入下一 agent task（等价 KVCOMM `spatial_info`）
5. 从 `diagnostics.timeline` 读取 `model.call.completed.timeToFirstByteMs`（probe agent 默认 index 2）

## 通信验证（L1–L3）

| 层级 | 字段 | 说明 |
|------|------|------|
| L1 | `child_session_key` 含 `:subagent:` | spawn 栈通路 |
| L2 | `task_includes_upstream` | 下游 task 是否包含上游 output |
| L3 | `--negative-control NC-1` | 故意去掉 agent_0 注入，对比 TTFT/输出 |

## 输出

- `results/O0-pre-A_<timestamp>.jsonl`：每条 agent 一行 + run_summary（行内 `ttft_ms` 仍为毫秒）
- `results/O0-pre-A_<timestamp>.summary.json`：汇总统计，**TTFT 均为秒（`_s` 后缀）**；`by_agent["0"|"1"|"2"].ttft_avg_s` 为各 agent 平均 TTFT；`probe` 为 scenario 中 probe agent（默认 agent 2）的 p50/p99/avg

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OPENCLAW_GATEWAY_URL` | `ws://127.0.0.1:18789` | Gateway WS |
| `OPENCLAW_GATEWAY_TOKEN` | | 认证 token |
| `OPENCLAW_DIAGNOSTICS_TIMELINE_PATH` | | TTFT 真源 |
| `BENCH_AGENT_ID` | `main` | Orchestrator agent |
| `BENCH_MODEL` | | Subagent 模型 |
| `BENCH_OUTPUT` | | 结果文件 basename（同 `--output`） |
| `COPY_PREFIX_REPEATS` | `64` | Copy 域 prefix：`" Ω"` 重复次数（对齐 KVCOMM `IN_LENGTH` 时请设 `512`） |
| `COPY_OUT_LENGTH` | `128` | Copy 输出目标（对齐 KVCOMM `OUT_LENGTH` 时请设 `512`） |
| `SEED` | （fixture 内固定 42） | Task 序列见 `fixtures/kvcomm_tasks_seed42.json` |

## 迁入 ClawBench

本 spike 逻辑对应计划 §9.2a.0c 中的 `kvcomm_runner.py` + `ttft_collector.py`；验证通过后可迁入 ClawBench `tasks-kvcomm/` lane。

## ClawBench Chain 能力 lane（Phase 2）

固定 **3-agent Chain** + ClawBench 原生任务（如 `t1-fs-quick-note`），在链末用 ClawBench `score_task_run` 做 **completion / trajectory / behavior** 打分（与 `clawbench run` 同一 scorer）。

与 Phase 1（`bench run --task-profile clawbench`）的区别：Phase 1 为 **text-only** Chain，subagent 无 `edit`，无法通过 file verifier；Phase 2 为 **capability** lane，共享 workspace + subagent 工具。

### 前置配置（顺序重要）

```bash
# ① ClawBench 原生：全局 tools.profile=coding
bash /src/clawbench/scripts/setup_vllm_clawbench.sh

# ② KVCOMM capability profile：main=minimal+sessions_spawn，subagent 有 read/edit/exec
cd /src/KVCOMM/openclaw && ./scripts/setup-openclaw.sh clawbench-capability

# ③ 冷重启 Gateway
openclaw gateway stop || true
openclaw gateway run
```

### 运行

```bash
cd /src/KVCOMM/openclaw

node cli.mjs bench run-clawbench \
  --agent-count 3 \
  --measure-runs 3 \
  --task-id t1-fs-quick-note \
  --dataset /src/KVCOMM/experiments/bench/datasets/tier1_clawbench.jsonl \
  --model vllm/Qwen3-32B \
  --output clawbench_chain_quicknote
```

输出：

- `results/<output>.jsonl` — 每 agent 的 TTFT / spawn 元数据（机器可读，每行一条 JSON）
- `results/<output>.summary.json` — 聚合统计（TTFT、ClawBench 分数）
- `results/<output>.formatted.txt` — **逐条记录格式化**（每个 key-value 独占一行，便于人工查看）
- `results/<output>.report.txt` — **聚合报告**（表格：TTFT、sidecar 路由、C/T/B 分数）
- run summary 行含 `capability_score`

对已有 jsonl 重新生成报告：

```bash
node scripts/format-bench-report.mjs results/clawbench-quicknote-kvreuse-005.jsonl
```

Dry-run（无需 Gateway）：

```bash
node cli.mjs bench run-clawbench --dry-run --task-id t1-fs-quick-note --agent-count 3
```

### 从 ClawBench CLI 调用

```bash
cd /src/clawbench
PYTHONPATH=/src/clawbench uv run clawbench kvcomm run-clawbench \
  --task-id t1-fs-quick-note --agent-count 3 --measure-runs 3 \
  --dataset /src/KVCOMM/experiments/bench/datasets/tier1_clawbench.jsonl \
  --model vllm/Qwen3-32B \
  --output clawbench_chain_quicknote
```

数据集 [`datasets/tier1_clawbench.jsonl`](datasets/tier1_clawbench.jsonl) 每行含 `capability_agent_tasks`（Chain 分工 prompt）与 `clawbench_ref.asset_packs`（workspace fixture）。

**Capability lane 说明**：`tools.invoke(sessions_spawn)` 默认不向 subagent 继承 `read`/`edit`/`write`。driver 会在 spawn 后通过 `sessions.patch` + `sessions.reset` + `sessions.send` 重跑子 agent，并在打分前 `syncCapabilityWorkspaceArtifacts` 将笔记同步到 chain workspace。`agent_1` 模板使用 `{{workspace_dir}}` 绝对路径写文件。
