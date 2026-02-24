# Completions Compatibility Proxy

这个服务提供老接口 `POST /v1/completions`，并转发到上游 `POST /v1/responses`。

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
