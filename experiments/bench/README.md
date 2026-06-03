# OpenClaw Subagent + TTFT Bench Driver (O0-pre spike)

基于计划 §9.2a.0d 的最小 harness：通过 Gateway `tools.invoke(sessions_spawn)` 栈驱动 3-agent Chain，并采集 per-agent TTFT。

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
--model /src/Meta-Llama-3.1-8B-Instruct
```

若使用本地 Llama，需先在 OpenClaw 中配置 vLLM/OpenAI-compat provider，再传 `vllm/...` 或对应 alias。

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
"vllm/Meta-Llama-3.1-8B-Instruct": {
  "params": { "extra_body": { "tool_choice": "none" } }
}
```

3. **Agent 工具 profile**：orchestrator 需要 `sessions_spawn`，必须用 **`coding`** 或 **`full`**。`minimal` 不含该工具，会报 `Tool not available: sessions_spawn`。

4. **Subagent 工具对齐（L1 限制）**：OpenClaw leaf subagent 若通过 `deny` 禁用全部 action 工具，会与 `gateway.tools.allow` 传入的 inherited allowlist 冲突，报 `No callable tools remain`。当前可用配置：

```json
"tools": {
  "profile": "coding",
  "subagents": {
    "tools": { "allow": ["session_status"] }
  }
}
```

`allow: ["session_status"]` 使 subagent 仅暴露一个无害工具（allow-only 过滤），禁用 web_search/update_plan 等 action 工具。模型仍可能输出 session_status JSON 文本；`tool_choice: none` + 纯 Copy task + 强化 `copy_role` 可降低概率。结果 JSONL 含 `tool_json_detected` / `copy_char_ratio` / `output_format_ok`。

修改 `openclaw.json` 后须 **冷重启 Gateway**（`openclaw gateway stop` → `openclaw gateway run`），热加载对 subagent 策略不可靠。

5. **vLLM 服务端要求**（常见报错）：

| 报错 | 原因 | 处理 |
|------|------|------|
| `sessions_spawn is blocked` | 未配置 `gateway.tools.allow` | 见上，重启 Gateway |
| `Tool not available: sessions_spawn` | `tools.profile: minimal` 且无 `alsoAllow` | 改 `profile: coding`，冷重启 Gateway |
| `model not allowed` | 未在 `agents.defaults.models` 注册 | 添加 `vllm/<model-id>` |
| `tool choice requires --enable-auto-tool-choice` | vLLM 未开 tool calling | bench 用 `tool_choice: none`（已配） |
| `No callable tools remain` | `subagents.tools.allow` 与 inherited allowlist 冲突 | 改用 `subagents.tools.deny` 禁用 action 工具，勿用 empty/minimal allow |

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
vllm serve /models/llama-3.1-8b-instruct \
  --served-model-name Meta-Llama-3.1-8B-Instruct \
  --dtype float16 \
  --max-model-len 16384 \
  --port 8001
```

OpenClaw 侧：`tools.profile: coding`，`contextWindow: 16384`，`compaction.reserveTokensFloor: 0`。

验证：`curl -s http://<vllm-host>:8001/v1/models` 中 `max_model_len` ≥ 16384。

## 快速开始

```bash
cd KVCOMM/experiments/bench
npm install

# 仅验证 dataset/scenario 渲染（无需 Gateway）
npm run dry-run

# 跑 1 条 task
npm run run -- --runs 1 --task-id micro-001 --model vllm/your-model

# 跑完整 tier0（每条 task 30 次）
npm run run -- --runs 30
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

- `results/O0-pre-A_<timestamp>.jsonl`：每条 agent 一行 + run_summary
- `results/O0-pre-A_<timestamp>.summary.json`：p50/p99/fallback 率

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OPENCLAW_GATEWAY_URL` | `ws://127.0.0.1:18789` | Gateway WS |
| `OPENCLAW_GATEWAY_TOKEN` | | 认证 token |
| `OPENCLAW_DIAGNOSTICS_TIMELINE_PATH` | | TTFT 真源 |
| `BENCH_AGENT_ID` | `main` | Orchestrator agent |
| `BENCH_MODEL` | | Subagent 模型 |
| `COPY_PREFIX_REPEATS` | `64` | Copy 域 prefix 长度（调试用） |
| `COPY_OUT_LENGTH` | `128` | Copy 输出 token 预算 |

## 迁入 ClawBench

本 spike 逻辑对应计划 §9.2a.0c 中的 `kvcomm_runner.py` + `ttft_collector.py`；验证通过后可迁入 ClawBench `tasks-kvcomm/` lane。
