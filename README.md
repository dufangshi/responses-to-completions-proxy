# Responses / Completions Compatibility Proxy

一个 OpenAI 兼容代理，统一提供：
- `POST /v1/files`
- `GET /v1/files`
- `GET /v1/files/{file_id}`
- `GET /v1/files/{file_id}/content`
- `DELETE /v1/files/{file_id}`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`
- `POST /v1/message`（`/v1/messages` 的兼容别名）
- `GET /v1/models`
- `GET /v1/models/{model_id}`

当前版本只支持单一上游。

现在仓库还内置了一个网页版聊天前端：
- Web UI：基于 `NextChat`
- 默认通过 Docker Compose 与 proxy 一起启动
- Web UI 只走本地 proxy，不直连真实上游
- Web UI 默认只展示一个模型，名称与 `DEFAULT_UPSTREAM_MODEL` 一致
- 即使前端请求里带了 `model`，proxy 仍会忽略并使用自己的默认模型

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

`setup` / `config` 现在默认只问 5 个问题：
- Web UI 端口 `WEB_PORT`
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

Web UI 默认地址：`http://127.0.0.1:38080`

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
WEB_PORT=38080
WEB_CODE=
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

# 单一 fast 开关:
# messages 上游 -> speed=fast
# responses 上游 -> service_tier=priority
DEFAULT_UPSTREAM_SPEED=fast
ENABLE_GENERIC_CHAT_SESSION_INFERENCE=true

# 调试日志
RAW_IO_LOG_ENABLED=false
RAW_IO_LOG_PATH=logs/raw_io.jsonl
RAW_IO_LOG_MAX_CHARS=120000
RAW_IO_LOG_KEEP_REQUESTS=10
```

> 说明：
> - `WEB_PORT` 是内置聊天网页的对外端口
> - `WEB_CODE` 留空表示 Web UI 不启用访问码；填写后会启用 NextChat 自带的访问码页
> - `UPSTREAM_BASE_URL` 可以写根域名，也可以直接写到 `/v1`
> - 代理内部会统一规范到 `/v1`
> - `UPSTREAM_MODE=messages` 时，请求会发到 `/v1/messages`
> - `UPSTREAM_MODE=responses` 时，请求会发到 `/v1/responses`
> - 默认 `UPSTREAM_STREAMING_ENABLED=true`，也就是即使客户端发普通非流式请求，proxy 也会优先用上游流式并在本地聚合结果
> - 代理会忽略下游传来的 `model`、`reasoning`、`speed`
> - 实际使用的模型、推理强度、fast 模式统一由 `.env` 控制
> - 默认模型是 `gpt-5.4`
> - 默认推理强度是 `medium`
> - Web UI 默认调用 `POST /v1/chat/completions`
> - Web UI 会隐藏用户自定义 API Key 输入，并把可选模型列表压缩为单一 `DEFAULT_UPSTREAM_MODEL`
> - `DEFAULT_UPSTREAM_SPEED` 是统一的 fast 开关
> - 设为 `fast` 时：
>   - `UPSTREAM_MODE=messages` -> 上游发送 `speed=fast`
>   - `UPSTREAM_MODE=responses` -> 上游发送 `service_tier=priority`
> - 留空或设为其他值时，不启用 fast 行为
> - `ENABLE_GENERIC_CHAT_SESSION_INFERENCE=true` 时，`/v1/chat/completions` 会为无显式 session id 的下游做保守的会话推断，并尽量稳定 `prompt_cache_key` 与历史前缀；判断不了时自动回退为普通转发

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
- `POST /v1/files`
  - OpenAI Files 兼容入口
- `GET /v1/files`
- `GET /v1/files/{file_id}`
- `GET /v1/files/{file_id}/content`
- `DELETE /v1/files/{file_id}`
- `GET /v1/models`
- `GET /v1/models/{model_id}`

所有主要 POST 入口都支持 `stream: true`。

---

## 内置 Web UI

项目自带一个 `NextChat` 前端快照，源码位于 `third_party/nextchat`。

默认部署方式：
- `completions-proxy`：`18010`
- `nextchat-web`：`38080`

启动后你可以直接访问：
- Proxy API：`http://127.0.0.1:18010/v1`
- Chat Web UI：`http://127.0.0.1:38080`

前端接入方式：
- Web UI 内部通过 Docker 网络访问 `http://completions-proxy:18010`
- 前端请求走 `POST /v1/chat/completions`
- 前端页面默认隐藏用户自定义 API Key 输入
- 前端模型列表默认只显示一个模型：`DEFAULT_UPSTREAM_MODEL`

如果你设置了：

```env
WEB_CODE=your-password
```

那么打开 Web UI 时会先进入访问码页。

## 四个主要入口分别做什么

- `POST /v1/responses`：新 SDK 推荐入口（原生 Responses）
- `POST /v1/chat/completions`：兼容 chat 客户端（含工具调用）
- `POST /v1/completions`：兼容老 prompt 接口
- `POST /v1/messages`：兼容 Claude / Anthropic Messages 风格客户端
- `POST /v1/files`：兼容 OpenAI SDK `client.files.create(...)`

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

### 5) `/v1/files`

上传文件：

```bash
curl -sS http://127.0.0.1:18010/v1/files \
  -H "Authorization: Bearer test-local" \
  -F "purpose=user_data" \
  -F "file=@/path/to/paper.pdf" | python -m json.tool
```

查询文件：

```bash
curl -sS http://127.0.0.1:18010/v1/files/file-xxxxxxxx | python -m json.tool
```

读取原始文件内容：

```bash
curl -L http://127.0.0.1:18010/v1/files/file-xxxxxxxx/content -o downloaded.pdf
```

说明：
- proxy 不会把文件上传到上游 `/v1/files`
- 文件会先保存在 proxy 本地，再在后续请求里自动展开成上游可接受的 `file_data`
- 当前最适合的用途是 PDF 输入

---

## OpenAI Python SDK（本地指向 proxy）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18010/v1", api_key="test-local")

for model in ["gpt-5.4", "claude-opus-4-6"]:
    r = client.responses.create(model=model, input="hello")
    print("requested =", model, "actual =", r.model, "text =", r.output_text)
```

### OpenAI Python SDK：先上传 PDF，再在请求里引用 `file_id`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18010/v1", api_key="test-local")

with open("paper.pdf", "rb") as f:
    uploaded = client.files.create(file=f, purpose="user_data")

response = client.responses.create(
    model="gpt-5.4",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize this PDF."},
                {"type": "input_file", "file_id": uploaded.id},
            ],
        }
    ],
)

print(uploaded.id)
print(response.output_text)
```

### OpenAI Python SDK：`chat.completions` + `file_id`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18010/v1", api_key="test-local")

with open("paper.pdf", "rb") as f:
    uploaded = client.files.create(file=f, purpose="user_data")

chat = client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this PDF."},
                {"type": "input_file", "file_id": uploaded.id},
            ],
        }
    ],
)

print(chat.choices[0].message.content)
```

### Anthropic `/v1/messages`：`document.source.type=file`

```bash
FILE_ID="file-xxxxxxxx"

curl -sS http://127.0.0.1:18010/v1/messages \
  -H 'x-api-key: test-local' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\":\"ignored-by-proxy\",
    \"max_tokens\":256,
    \"messages\":[
      {
        \"role\":\"user\",
        \"content\":[
          {\"type\":\"text\",\"text\":\"Summarize this PDF.\"},
          {\"type\":\"document\",\"title\":\"paper.pdf\",\"source\":{\"type\":\"file\",\"file_id\":\"${FILE_ID}\"}}
        ]
      }
    ]
  }" | python -m json.tool
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

会同时启动 proxy 和内置 Web UI：

```bash
docker rm -f completions-proxy nextchat-web 2>/dev/null || true
docker compose rm -sf completions-proxy nextchat-web
docker compose up -d --build completions-proxy nextchat-web
```

### docker run

如果你只手动跑 proxy，可以继续这样：

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
  -v "$(pwd)/data:/app/data" \
  ghcr.io/dufangshi/responses-to-completions-proxy:latest
```

补充：
- `./data` 用来持久化 `/v1/files` 上传的本地文件
- 如果你删掉这个目录，之前返回过的 `file_id` 就会失效
- 如果你也要手动跑 Web UI，推荐直接使用 `docker compose`，因为前端需要和 proxy 处于同一个 Docker 网络中

---

## 代码入口

- `/v1/files`：`app/routes/files.py`
- `/v1/completions`：`app/routes/completions.py`
- `/v1/chat/completions`：`app/routes/chat_completions.py`
- `/v1/responses`：`app/routes/responses.py`
- 路由网关：`app/services/responses_client.py`
- Antigravity 适配：`app/services/antigravity_adapter.py`
