# Responses / Completions Compatibility Proxy

一个 OpenAI 兼容代理，统一提供：
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`
- `POST /v1/message`（`/v1/messages` 的兼容别名）
- `GET /v1/models`
- `GET /v1/models/{model_id}`

当前版本只支持单一上游。

代理对外同时兼容：
- OpenAI SDK / Responses API
- OpenAI Chat Completions API
- OpenAI legacy Completions API
- Anthropic Messages 风格请求

你可以在配置里切换单一上游的请求模式：
- `UPSTREAM_MODE=responses`：上游走 `POST /v1/responses`
- `UPSTREAM_MODE=messages`：上游走 `POST /v1/messages`

---

## 推荐用法：统一管理脚本

仓库根目录提供了一个入口脚本：`./proxy.sh`

常用命令：

```bash
./proxy.sh setup
./proxy.sh config
./proxy.sh up
./proxy.sh down
./proxy.sh logs
./proxy.sh status
./proxy.sh update
```

- `setup`：首次安装，交互式写入 `.env`，然后启动容器
- `config`：重新输入配置，自动备份旧 `.env`，然后重建容器
- `update`：在工作区干净时自动 `git pull --rebase --autostash`，然后重建容器

`setup` / `config` 现在默认只问 4 个问题：
- 上游 `Base URL`
- 上游 `API Key`
- 上游模式：`responses` 或 `messages`
- 默认模型 `DEFAULT_UPSTREAM_MODEL`

高级项不再交互询问，直接保留默认值或现有 `.env` 里的值。需要时手动编辑 `.env`。

---

## 一键启动

```bash
chmod +x ./proxy.sh
./proxy.sh setup
```

服务默认地址：`http://127.0.0.1:18010`

---

## 重配与更新

重新输入配置：

```bash
./proxy.sh config
```

提示：
- 直接回车：保留当前值
- 输入 `-`：清空当前值（适合 `FORCE_UPSTREAM_MODEL` 这类可选项）

拉最新代码并重建容器：

```bash
./proxy.sh update
```

查看状态和日志：

```bash
./proxy.sh status
./proxy.sh logs
```

---

## 环境变量（高级）

最小必填项：

```env
UPSTREAM_BASE_URL=https://sub.lnz-study.com
UPSTREAM_MODE=responses
UPSTREAM_API_KEY=sk-xxx
DEFAULT_UPSTREAM_MODEL=gpt-5.4
```

> 注意：仓库不会提交你的 `.env`（`.gitignore` 已忽略 `.env`），所以拉取最新代码不会自动改你服务器上的 `.env`。
>
> 如果你使用 `./proxy.sh setup` 或 `./proxy.sh config`，脚本会自动写入 `.env`，一般不需要手工编辑。

其余常用项：

```env
APP_HOST=127.0.0.1
APP_PORT=18010
UPSTREAM_MESSAGES_API_VERSION=2023-06-01
UPSTREAM_TIMEOUT_SECONDS=120
UPSTREAM_STREAMING_ENABLED=true

# 限流与降级
UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS=10
UPSTREAM_FALLBACK_MODEL=

# 强制模型链
USE_FORCE_MODEL=false
FORCE_UPSTREAM_MODEL=
DEFAULT_UPSTREAM_MODEL=gpt-5.4

# 对 gpt-5.3-codex / gpt-5.4 / claude-opus-4-6 / claude-sonnet-4-6 / claude-opus-4-5 生效
DEFAULT_REASONING_EFFORT=medium

# 仅 messages 上游模式生效
DEFAULT_UPSTREAM_SPEED=fast

# 调试日志
RAW_IO_LOG_ENABLED=false
RAW_IO_LOG_PATH=logs/raw_io.jsonl
RAW_IO_LOG_MAX_CHARS=120000
RAW_IO_LOG_KEEP_REQUESTS=10
```

> 说明：
> - `UPSTREAM_BASE_URL` 可以写根域名，也可以直接写到 `/v1`
> - 代理内部会统一规范到 `/v1`
> - `UPSTREAM_MODE=messages` 时，请求会发到 `/v1/messages`
> - `UPSTREAM_MODE=responses` 时，请求会发到 `/v1/responses`
> - 默认 `UPSTREAM_STREAMING_ENABLED=true`，也就是即使客户端发普通非流式请求，proxy 也会优先用上游流式并在本地聚合结果
> - 代理会忽略下游传来的 `model`、`reasoning`、`speed`
> - 实际使用的模型、推理强度、fast 模式统一由 `.env` 控制
> - 默认模型是 `gpt-5.4`
> - 默认推理强度是 `medium`
> - 默认 `DEFAULT_UPSTREAM_SPEED=fast`，且只在 `UPSTREAM_MODE=messages` 时生效

### 旧 `.env` 键名迁移（从旧版本升级时）

若你之前使用的是双上游版本，推荐直接重新执行：

```bash
./proxy.sh config
```

旧版的双上游变量不再是推荐配置方式。

---

## 上游模式切换

模式由 `UPSTREAM_MODE` 控制：

- `responses`
  - 上游请求：`POST /v1/responses`
  - 适合 OpenAI Responses 风格上游
- `messages`
  - 上游请求：`POST /v1/messages`
  - 适合 Claude Messages 风格上游

模型、reasoning effort、fast/speed 都由 proxy 内部配置统一控制，不再信任下游传参。

---

## 下游支持的请求格式

- `POST /v1/responses`
  - OpenAI Responses 原生入口
- `POST /v1/chat/completions`
  - OpenAI Chat Completions 兼容入口
- `POST /v1/completions`
  - OpenAI 老式 prompt/completions 兼容入口
- `POST /v1/messages`
  - Anthropic Messages 风格入口
- `POST /v1/message`
  - `POST /v1/messages` 的兼容别名
- `GET /v1/models`
- `GET /v1/models/{model_id}`

所有主要 POST 入口都支持 `stream: true`。

## 四个主要入口分别做什么

- `POST /v1/responses`：新 SDK 推荐入口（原生 Responses）
- `POST /v1/chat/completions`：兼容 chat 客户端（含工具调用）
- `POST /v1/completions`：兼容老 prompt 接口
- `POST /v1/messages`：兼容 Claude / Anthropic Messages 风格客户端

---

## 调用示例

### 1) `/v1/responses`

```bash
curl -sS http://127.0.0.1:18010/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.3-codex","input":"hello"}' | python -m json.tool
```

流式：

```bash
curl -N http://127.0.0.1:18010/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-opus-4-6","input":"hello","stream":true}'
```

### 2) `/v1/chat/completions`

```bash
curl -sS http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-3.1-pro-high","messages":[{"role":"user","content":"hello"}]}' | python -m json.tool
```

### 3) `/v1/completions`

```bash
curl -sS http://127.0.0.1:18010/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.3-codex","prompt":"hello"}' | python -m json.tool
```

### 4) `/v1/messages`

```bash
curl -sS http://127.0.0.1:18010/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"anything-here-is-ignored",
    "max_tokens":128,
    "messages":[
      {
        "role":"user",
        "content":[{"type":"text","text":"Reply with hello only."}]
      }
    ]
  }' | python -m json.tool
```

流式：

```bash
curl -N http://127.0.0.1:18010/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"anything-here-is-ignored",
    "messages":[{"role":"user","content":"Reply with hello only."}],
    "stream":true
  }'
```

工具调用说明：
- 当下游传入 Anthropic 风格 `tool_choice: {"type":"tool","name":"..."}` 时，proxy 会自动把工具列表裁成该工具，并向当前 Claude 风格上游发送等价的 `tool_choice: {"type":"any"}`，以兼容该上游的参数限制。

---

## OpenAI Python SDK（本地指向 proxy）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18010/v1", api_key="test-local")

for model in ["gpt-5.4", "claude-opus-4-6"]:
    r = client.responses.create(model=model, input="hello")
    print("requested =", model, "actual =", r.model, "text =", r.output_text)
```

---

## OpenClaw 对接（Custom Provider）

在 OpenClaw 中选择 Custom Provider：
- API 类型：`openai-completions`
- Base URL：`http://127.0.0.1:18010/v1`
- API Key：任意占位值（例如 `111`）
- Model：随便填一个占位模型名也行，但建议和 `.env` 里的 `DEFAULT_UPSTREAM_MODEL` 保持一致

然后手动检查 `openclaw.json` 里的模型参数（重要）：
- `contextWindow` 设大（建议 `400000`）
- `maxTokens` 设大（建议 `128000`）

---

## 手动 Docker 命令（如果你不用脚本）

如果你修改了 `.env`，要注意：
- `docker compose up -d --build` 不一定会刷新运行中容器的环境变量
- 最稳妥的方式是先删掉服务容器，再重新创建

### docker compose

```bash
docker rm -f completions-proxy 2>/dev/null || true
docker compose rm -sf completions-proxy
docker compose up -d --build completions-proxy
```

### docker run

```bash
docker pull ghcr.io/dufangshi/responses-to-completions-proxy:latest
docker rm -f completions-proxy || true
docker run -d --name completions-proxy \
  --restart unless-stopped \
  --env-file .env \
  -e APP_HOST=0.0.0.0 \
  -e APP_PORT=18010 \
  -p 18010:18010 \
  -v "$(pwd)/logs:/app/logs" \
  ghcr.io/dufangshi/responses-to-completions-proxy:latest
```

---

## 代码入口

- `/v1/completions`：`app/routes/completions.py`
- `/v1/chat/completions`：`app/routes/chat_completions.py`
- `/v1/responses`：`app/routes/responses.py`
- 路由网关：`app/services/responses_client.py`
- Antigravity 适配：`app/services/antigravity_adapter.py`
