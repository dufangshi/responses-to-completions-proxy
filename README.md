# Completions Compatibility Proxy

这个服务提供老接口 `POST /v1/completions`，并转发到上游 `POST /v1/responses`。
支持：
- `/v1/completions` 和 `/completions`
- `/v1/chat/completions` 和 `/chat/completions`
- `stream=false` 与 `stream=true`

## 一键启动（Docker）

先准备 `.env`（至少填好 `UPSTREAM_BASE_URL` 和 `UPSTREAM_API_KEY`），然后执行：

```bash
docker compose up -d --build
```

启动后服务地址：`http://127.0.0.1:18010`

## 一键启动（Public Git Package / GHCR）

仓库配置了自动发布到 GHCR（`ghcr.io/dufangshi/responses-to-completions-proxy`）。

```bash
docker run -d --name completions-proxy \
  --env-file .env \
  -p 18010:18010 \
  ghcr.io/dufangshi/responses-to-completions-proxy:latest
```

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
