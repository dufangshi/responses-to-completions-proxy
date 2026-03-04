# Responses / Completions Compatibility Proxy

一个 OpenAI 兼容代理，统一提供：
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

并按模型自动路由上游：
- OpenAI 路由：`/v1/responses`
- Antigravity 路由：`/antigravity/v1/messages`（Gemini / Claude）

---

## 一键启动（Docker）

```bash
docker compose up -d --build
curl -sS http://127.0.0.1:18010/healthz
```

服务默认地址：`http://127.0.0.1:18010`

---

## 环境变量（精简）

只需要这 3 个必填项：

```env
UPSTREAM_BASE_URL=https://sub.lnz-study.com
UPSTREAM_OPENAI_API_KEY=sk-xxx
UPSTREAM_ANTIGRAVITY_API_KEY=sk-xxx
```

> 注意：仓库不会提交你的 `.env`（`.gitignore` 已忽略 `.env`），所以拉取最新代码不会自动改你服务器上的 `.env`。

其余常用项：

```env
APP_HOST=127.0.0.1
APP_PORT=18010
UPSTREAM_TIMEOUT_SECONDS=120

# 限流与降级
ANTIGRAVITY_MIN_REQUEST_INTERVAL_SECONDS=10
ANTIGRAVITY_FALLBACK_MODEL=gemini-3-flash-preview

# 可选：覆盖自动拼接地址
# UPSTREAM_OPENAI_BASE_URL=https://sub.lnz-study.com/v1
# UPSTREAM_ANTIGRAVITY_BASE_URL=https://sub.lnz-study.com/antigravity

# 强制模型链
USE_FORCE_MODEL=false
FORCE_UPSTREAM_MODEL=
DEFAULT_UPSTREAM_MODEL=gpt-5.3-codex

# 仅 gpt-5.3-codex 生效
DEFAULT_REASONING_EFFORT=high

# 哪些前缀走 OpenAI 路由（其余都走 Antigravity）
OPENAI_MODEL_PREFIXES=gpt-,o1,o3,o4,text-embedding,text-moderation,whisper,tts,dall-e,omni

# 调试日志
RAW_IO_LOG_ENABLED=false
RAW_IO_LOG_PATH=logs/raw_io.jsonl
RAW_IO_LOG_MAX_CHARS=120000
RAW_IO_LOG_KEEP_REQUESTS=10
```

> 说明：`UPSTREAM_BASE_URL` 只需填根域名，代理会自动拼接：
> - OpenAI：`/v1`
> - Antigravity：`/antigravity`

### 旧 `.env` 键名迁移（从旧版本升级时）

若你之前使用的是旧键名，可按下面方式迁移到新键名：

```bash
# 备份
cp .env .env.bak.$(date +%Y%m%d-%H%M%S)

# 旧键名 -> 新键名
sed -i '' 's/^UPSTREAM_API_KEY=/UPSTREAM_OPENAI_API_KEY=/' .env
sed -i '' 's/^UPSTREAM_GEMINI_API_KEY=/UPSTREAM_ANTIGRAVITY_API_KEY=/' .env
sed -i '' 's/^GEMINI_MIN_REQUEST_INTERVAL_SECONDS=/ANTIGRAVITY_MIN_REQUEST_INTERVAL_SECONDS=/' .env
sed -i '' 's/^GEMINI_FALLBACK_MODEL=/ANTIGRAVITY_FALLBACK_MODEL=/' .env
```

如果是 Linux（GNU sed），把 `sed -i ''` 改为 `sed -i`。

兼容性说明：当前版本不兼容旧键名；若仍使用旧键名，服务会在启动时直接报错退出。

---

## 自动路由规则

1. 先解析 `model`（支持 `gpt-5.3-codex:high` 这种写法）
2. 若命中 `OPENAI_MODEL_PREFIXES`，走 OpenAI 上游 `/v1/responses`
3. 否则走 Antigravity `/v1/messages`

当前典型结果：
- `gpt-5.3-codex` → OpenAI
- `gemini-3.1-pro-high` → Antigravity
- `claude-opus-4-6` → Antigravity

---

## 三个入口分别做什么

- `POST /v1/responses`：新 SDK 推荐入口（原生 Responses）
- `POST /v1/chat/completions`：兼容 chat 客户端（含工具调用）
- `POST /v1/completions`：兼容老 prompt 接口

三个入口都支持 `stream: true`。

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

---

## OpenAI Python SDK（本地指向 proxy）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18010/v1", api_key="test-local")

for model in ["gpt-5.3-codex", "gemini-3.1-pro-high", "claude-opus-4-6"]:
    r = client.responses.create(model=model, input="hello")
    print(model, r.output_text)
```

---

## OpenClaw 对接（Custom Provider）

在 OpenClaw 中选择 Custom Provider：
- API 类型：`openai-completions`
- Base URL：`http://127.0.0.1:18010/v1`
- API Key：任意占位值（例如 `111`）
- Model：`gpt-5.3-codex`

然后手动检查 `openclaw.json` 里的模型参数（重要）：
- `contextWindow` 设大（建议 `400000`）
- `maxTokens` 设大（建议 `128000`）

---

## 容器镜像更新与重启

### 服务器侧一键更新（推荐）

```bash
cd /path/to/responses-to-completions-proxy
git pull --rebase origin main
docker compose pull completions-proxy
# 处理同名旧容器冲突（例如之前用 docker run 创建过）
docker rm -f completions-proxy 2>/dev/null || true
docker compose up -d --no-build completions-proxy
docker compose ps
curl -sS http://127.0.0.1:18010/healthz
```

### docker compose

```bash
docker compose pull completions-proxy
docker compose up -d --no-build completions-proxy
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
