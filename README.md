# Completions Compatibility Proxy

这个服务提供老接口 `POST /v1/completions`，并转发到上游 `POST /v1/responses`。
支持：
- `/v1/completions` 和 `/completions`
- `/v1/chat/completions` 和 `/chat/completions`
- `stream=false` 与 `stream=true`

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
