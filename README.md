# Completions Compatibility Proxy

这个服务提供老接口 `POST /v1/completions`，并统一兼容转发到上游：
- OpenAI 上游（`/responses`）
- Gemini 上游（`/models/{model}:generateContent` / `streamGenerateContent`）
支持：
- `/v1/completions` 和 `/completions`
- `/v1/chat/completions` 和 `/chat/completions`
- `/v1/responses` 和 `/responses`
- `/v1/models` 和 `/models`
- `stream=false` 与 `stream=true`

## 快速跳转

- [三种入口对应关系](#api-entry-map)
- [`/v1/completions` 详细调用](#api-completions)
- [`/v1/chat/completions` 详细调用](#api-chat-completions)
- [`/v1/responses` 详细调用（OpenAI SDK）](#api-responses)
- [容器镜像更新与重启](#docker-upgrade)

## 一键启动（Docker）

先准备 `.env`（至少填好 `UPSTREAM_BASE_URL` 和 `UPSTREAM_API_KEY`），然后执行：

```bash
docker compose up -d --build
```

启动后服务地址：`http://127.0.0.1:18010`

<a id="api-entry-map"></a>
## 三种入口对应关系

| 入口 | 代码位置 | 主要用途 | 推荐使用场景 |
| --- | --- | --- | --- |
| `/v1/completions` | `app/routes/completions.py` | 兼容老式 Text Completions（`prompt`） | 旧客户端、仅文本续写 |
| `/v1/chat/completions` | `app/routes/chat_completions.py` | 兼容 Chat Completions（`messages`） | 现有 chat 客户端、工具调用 |
| `/v1/responses` | `app/routes/responses.py` | 直连 Responses API（新 SDK 原生） | OpenAI 官方 SDK（推荐） |

模型参数规则（三个入口通用）：
- 支持在 `model` 末尾附加推理强度（仅 `gpt-5.3-codex`）：`gpt-5.3-codex:low|medium|high|xhigh`
- 若解析后模型不是 `gpt-5.3-codex`，代理会自动移除 `reasoning` 参数，避免上游报错
- 若请求里不传 `model`（主要是 `/v1/responses`），会回退到 `DEFAULT_UPSTREAM_MODEL`

<a id="api-completions"></a>
## `/v1/completions` 详细调用

请求特点：
- 入参核心是 `prompt`
- 返回是 `choices[].text`
- 适合兼容老接口

示例（curl）：

```bash
curl -sS http://127.0.0.1:18010/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-local' \
  -d '{
    "model":"gpt-5.3-codex:high",
    "prompt":"hello",
    "max_tokens":64,
    "stream":false
  }' | python -m json.tool
```

流式示例：

```bash
curl -N http://127.0.0.1:18010/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-5.3-codex:medium",
    "prompt":"hello",
    "stream":true
  }'
```

<a id="api-chat-completions"></a>
## `/v1/chat/completions` 详细调用

请求特点：
- 入参核心是 `messages`
- 返回是 `choices[].message`
- 支持工具调用（`tools` / `tool_choice`）

示例（curl）：

```bash
curl -sS http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-local' \
  -d '{
    "model":"gpt-5.3-codex:high",
    "messages":[{"role":"user","content":"hello"}],
    "max_tokens":64,
    "stream":false
  }' | python -m json.tool
```

流式示例：

```bash
curl -N http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-5.3-codex:high",
    "messages":[{"role":"user","content":"hello"}],
    "stream":true
  }'
```

<a id="api-responses"></a>
## `/v1/responses` 详细调用（OpenAI SDK）

说明：
- OpenAI 官方 SDK 的 `responses.create(...)` 本质调用的是 `/responses`
- 这是当前推荐入口（与官方新接口一致）

### JavaScript / TypeScript

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:18010/v1",
  apiKey: "111",
});

const response = await client.responses.create({
  model: "gpt-5.3-codex:high",
  input: "Write a short bedtime story about a unicorn.",
});

console.log(response.output_text);
```

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:18010/v1",
    api_key="111",
)

response = client.responses.create(
    model="gpt-5.3-codex:high",
    input="Write a short bedtime story about a unicorn.",
)

print(response.output_text)
```

### 直接 HTTP 示例（不依赖 SDK）

```bash
curl -sS http://127.0.0.1:18010/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-local' \
  -d '{
    "model":"gpt-5.3-codex:high",
    "input":"Write a short bedtime story about a unicorn."
  }' | python -m json.tool
```

## 一键启动（Public Git Package / GHCR）

仓库配置了自动发布到 GHCR（`ghcr.io/dufangshi/responses-to-completions-proxy`）。

```bash
docker run -d --name completions-proxy \
  --env-file .env \
  -p 18010:18010 \
  ghcr.io/dufangshi/responses-to-completions-proxy:latest
```

<a id="docker-upgrade"></a>
## 容器镜像更新与重启

### 方式 A：docker compose（推荐）

在项目目录执行：

```bash
cd /path/to/chat_complete_proxy
docker compose pull completions-proxy
docker compose up -d --no-build completions-proxy
```

如果你需要固定到某个版本标签（例如 `v1.2.3`），建议使用下方“方式 B”并把镜像写成：
`ghcr.io/dufangshi/responses-to-completions-proxy:v1.2.3`。

### 方式 B：docker run 直接重建容器

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

更新后可用以下命令确认：

```bash
docker ps --filter name=completions-proxy
curl -sS http://127.0.0.1:18010/healthz
```

## OpenClaw 对接教程（Custom Provider）

如果你要把这个代理接到 OpenClaw，推荐走向导配置，不必手改大段 JSON：

1. 先启动本代理（本地监听 `127.0.0.1:18010`）
2. 打开配置向导：`openclaw configure`（或 `openclaw config`）
3. 进入 Model/Provider 相关步骤，选择 **Custom Provider**
4. Provider 名称填你自己的名字（例如 `custom-proxy-oai`）
5. API 类型选 `openai-completions`
6. Base URL 填 `http://127.0.0.1:18010/v1`
7. API Key 可填占位值（例如 `111`，由你的代理自行处理）
8. 添加模型 `gpt-5.3-codex`（显示名可自定义）
9. **务必手动把 `contextWindow` 和 `maxTokens` 调大**（见下一节建议值），否则容易因为默认值过小导致请求被截断或校验报错

可用以下命令快速确认模型是否可见：

```bash
openclaw models list
```

### 手动修改 `openclaw.json`（关键）

在你通过向导导入 Custom Provider 之后，建议再手动检查一次模型参数：

1. 打开 OpenClaw 配置文件 `openclaw.json`
2. 找到你刚添加的 Custom Provider（例如 `custom-proxy-oai`）
3. 在对应模型（例如 `gpt-5.3-codex`）下确认并修改：
   - `contextWindow` 为 `400000`
   - `maxTokens` 为 `128000`
4. 保存后重启 OpenClaw（或重开会话）使配置生效

原因：在 **Custom Provider 显式配置** 场景，OpenClaw 通常以本地模型配置为准；若值过小会导致上下文预算不足或参数校验失败。

## gpt-5.3-codex 推荐窗口参数

- `contextWindow`: `400000`
- `maxTokens`（最大输出）: `128000`

这两项建议与你在 OpenClaw Custom Provider 里保持一致（或至少不要低于你的实际业务需求）。

另外，代理现已提供 `/v1/models` 元数据（包含 `contextWindow/maxTokens`）。
即便如此，OpenClaw 在 **Custom Provider 显式配置** 场景下仍建议在本地模型配置里明确写入这两个值，避免不同版本行为差异导致的窗口预算不一致。

## 新设备最简步骤（默认已安装 uv）

```bash
cd /path/to/chat_complete_proxy
uv sync
```

## 配置

项目会自动读取根目录 `.env`。默认已经写好：
- `APP_HOST=127.0.0.1`
- `APP_PORT=18010`（不常用端口）
- `UPSTREAM_BASE_URL`
- `UPSTREAM_API_KEY`
- `UPSTREAM_GEMINI_BASE_URL`
- `UPSTREAM_GEMINI_API_KEY`
- `GEMINI_MIN_REQUEST_INTERVAL_SECONDS`（默认 `10`，Gemini 上游最小请求间隔，防止频率过高触发账号池限流）
- `GEMINI_FALLBACK_MODEL`（默认 `gemini-3-flash-preview`，当 Gemini 主模型返回 429/5xx 时自动回退）
- `DEFAULT_UPSTREAM_MODEL`
- `DEFAULT_REASONING_EFFORT`（默认 `high`，可选：`low` / `medium` / `high` / `xhigh`）
- `OPENAI_MODEL_PREFIXES`（默认：`gpt-,o1,o3,o4,text-embedding,text-moderation,whisper,tts,dall-e,omni`）
- `RAW_IO_LOG_ENABLED`（调试开关，`true` 时记录原始请求/响应）
- `RAW_IO_LOG_PATH`（默认 `logs/raw_io.jsonl`）
- `RAW_IO_LOG_MAX_CHARS`（每个字符串字段最大记录长度）
- `RAW_IO_LOG_KEEP_REQUESTS`（默认 `10`，仅保留最近 N 次请求的日志）

如果要改配置，直接编辑项目根目录 `.env`。

### OpenAI / Gemini 自动路由规则

- 代理先解析 `model`（包括你之前支持的 `gpt-5.3-codex:high` 这种推理后缀）
- 若模型前缀命中 `OPENAI_MODEL_PREFIXES`，走 `UPSTREAM_BASE_URL`（OpenAI 上游）
- 其他模型默认走 `UPSTREAM_GEMINI_BASE_URL`（Gemini 上游）
- 当目标模型不是 `gpt-5.3-codex` 时，会自动移除 `reasoning` 参数，避免上游报错
- 为避免客户端 `max_tokens/max_output_tokens` 配错，代理在 Gemini 常用模型上会做上限裁剪；若仍触发 `INVALID_ARGUMENT`，会自动去掉该参数重试
- `gpt-5.3-codex` 在当前上游兼容模式下不透传 `max_output_tokens`（避免上游 `Unsupported parameter` 报错）
- Gemini 请求默认做 10 秒节流（`GEMINI_MIN_REQUEST_INTERVAL_SECONDS`），可按需调小/关闭（设为 `0`）
- Gemini 主模型若触发 429/5xx，会自动 fallback 到 `GEMINI_FALLBACK_MODEL`

### 常用模型限额（代理内置）

- `gpt-5.3-codex`: `contextWindow=400000`，`maxTokens=128000`
- `gemini-3-flash-preview`: `contextWindow=1048576`，`maxTokens=65536`
- `gemini-3.1-pro-preview`: `contextWindow=1048576`，`maxTokens=65536`

这些值会用于 `/v1/models` 返回，并用于 Gemini 上游请求时的 `max_output_tokens` 防越界修正。

示例（Gemini）：

```bash
curl -sS http://127.0.0.1:18010/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-local' \
  -d '{"model":"gemini-3.1-pro-preview","input":"hello"}' | python -m json.tool
```

## 启动

```bash
uv run run.py
```

## Docker 发布说明

- 已包含 Docker 构建文件：`Dockerfile`
- 已包含本地编排：`docker-compose.yml`
- 已包含 GitHub Actions 自动发布：`.github/workflows/publish-ghcr.yml`
- `main` 分支推送会更新 `latest` 标签；推 `v*` tag 会发布同名版本标签
- 若 GHCR 包可见性未自动继承仓库公开属性，请在 GitHub Package 页面手动设为 **Public**

## 快速测试

```bash
curl -sS http://127.0.0.1:18010/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-local' \
  -d '{"model":"gpt-5.3-codex","prompt":"hello","max_tokens":64,"stream":false}' | python -m json.tool
```

```bash
curl -sS http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-local' \
  -d '{"model":"gpt-5.3-codex","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"stream":false}' | python -m json.tool
```

```bash
curl -N http://127.0.0.1:18010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.3-codex","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Gemini 兼容层回归测试

包含 Python SDK + Node SDK + 工具调用 + 流式 + 重试场景的本地集成测试：

```bash
python scripts/test_gemini_sdk_e2e.py
```
