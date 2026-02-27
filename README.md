# Completions Compatibility Proxy

这个服务提供老接口 `POST /v1/completions`，并转发到上游 `POST /v1/responses`。
支持：
- `/v1/completions` 和 `/completions`
- `/v1/chat/completions` 和 `/chat/completions`
- `/v1/responses` 和 `/responses`
- `/v1/models` 和 `/models`
- `stream=false` 与 `stream=true`

## 一键启动（Docker）

先准备 `.env`（至少填好 `UPSTREAM_BASE_URL` 和 `UPSTREAM_API_KEY`），然后执行：

```bash
docker compose up -d --build
```

启动后服务地址：`http://127.0.0.1:18010`

## OpenAI SDK 兼容（Responses API）

本代理已支持 OpenAI 官方 SDK 的 `responses.create` 调用路径（`POST /v1/responses`），可直接把 `baseURL` 指向本代理：

### JavaScript / TypeScript

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:18010/v1",
  apiKey: "111",
});

const response = await client.responses.create({
  model: "gpt-5.3-codex",
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
    model="gpt-5.3-codex",
    input="Write a short bedtime story about a unicorn.",
)

print(response.output_text)
```

说明：
- SDK 的 `responses.create` 本质调用的是 `/responses`（不是 `/chat/completions`）。
- 代理会接收并透传 `model` 参数；若你请求里不传 `model`，会自动回退到 `DEFAULT_UPSTREAM_MODEL`。
- 支持在 `model` 末尾附加推理强度（仅 `gpt-5.3-codex`）：例如 `gpt-5.3-codex:high` / `gpt-5.3-codex:medium`。
- 若解析后模型不是 `gpt-5.3-codex`，代理会自动移除 `reasoning` 参数，避免上游因不支持而报错。

## 一键启动（Public Git Package / GHCR）

仓库配置了自动发布到 GHCR（`ghcr.io/dufangshi/responses-to-completions-proxy`）。

```bash
docker run -d --name completions-proxy \
  --env-file .env \
  -p 18010:18010 \
  ghcr.io/dufangshi/responses-to-completions-proxy:latest
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
- `DEFAULT_UPSTREAM_MODEL`
- `DEFAULT_REASONING_EFFORT`（默认 `high`，可选：`low` / `medium` / `high` / `xhigh`）
- `RAW_IO_LOG_ENABLED`（调试开关，`true` 时记录原始请求/响应）
- `RAW_IO_LOG_PATH`（默认 `logs/raw_io.jsonl`）
- `RAW_IO_LOG_MAX_CHARS`（每个字符串字段最大记录长度）
- `RAW_IO_LOG_KEEP_REQUESTS`（默认 `10`，仅保留最近 N 次请求的日志）

如果要改配置，直接编辑项目根目录 `.env`。

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
