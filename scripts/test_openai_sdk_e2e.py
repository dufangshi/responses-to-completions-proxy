#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_TEST_DIR = Path("/tmp/openai-node-sdk-test")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http_ready(url: str, timeout_sec: float = 20.0) -> None:
    deadline = time.time() + timeout_sec
    last_error: str | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.5)
            if response.status_code < 500:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"Service not ready: {url}; last_error={last_error}")


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _extract_input_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    output = content_item.get("output")
                    if isinstance(output, str):
                        parts.append(output)
            elif isinstance(content, str):
                parts.append(content)
        return " ".join(parts).strip()
    return ""


def _write_mock_upstream_script(path: Path) -> None:
    source = textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import sys
        import time
        from typing import Any

        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse

        app = FastAPI()


        def extract_input_text(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if isinstance(content, list):
                        for content_item in content:
                            if not isinstance(content_item, dict):
                                continue
                            text = content_item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                            output = content_item.get("output")
                            if isinstance(output, str):
                                parts.append(output)
                    elif isinstance(content, str):
                        parts.append(content)
                return " ".join(parts).strip()
            return ""


        def build_response_object(payload: dict[str, Any]) -> dict[str, Any]:
            model = payload.get("model") or "mock-default-model"
            input_text = extract_input_text(payload.get("input"))
            reasoning = payload.get("reasoning")
            reasoning_effort = None
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
                if isinstance(effort, str):
                    reasoning_effort = effort
            text = f"mock:{model}:{input_text}:effort={reasoning_effort or 'none'}".strip(":")
            created = int(time.time())
            return {
                "id": f"resp_mock_{created}",
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": model,
                "output": [
                    {
                        "id": "msg_mock",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            }


        @app.post("/v1/responses")
        async def responses(request: Request):
            payload = await request.json()
            if not isinstance(payload, dict):
                return JSONResponse(status_code=400, content={"error": {"message": "bad request"}})

            response_object = build_response_object(payload)

            if payload.get("stream") is True:
                content_text = response_object["output"][0]["content"][0]["text"]
                split_at = max(1, len(content_text) // 2)
                part1 = content_text[:split_at]
                part2 = content_text[split_at:]

                async def iterator():
                    created_event = {
                        "type": "response.created",
                        "response": {
                            "id": response_object["id"],
                            "created_at": response_object["created_at"],
                            "model": response_object["model"],
                        },
                    }
                    yield f"event: response.created\\ndata: {json.dumps(created_event, ensure_ascii=False)}\\n\\n"
                    for delta in (part1, part2):
                        if not delta:
                            continue
                        delta_event = {
                            "type": "response.output_text.delta",
                            "delta": delta,
                            "output_index": 0,
                        }
                        yield f"event: response.output_text.delta\\ndata: {json.dumps(delta_event, ensure_ascii=False)}\\n\\n"
                    completed_event = {"type": "response.completed", "response": response_object}
                    yield f"event: response.completed\\ndata: {json.dumps(completed_event, ensure_ascii=False)}\\n\\n"
                    yield "data: [DONE]\\n\\n"

                return StreamingResponse(iterator(), media_type="text/event-stream")

            return JSONResponse(status_code=200, content=response_object)


        if __name__ == "__main__":
            port = int(sys.argv[1])
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
        """
    ).strip()
    path.write_text(source, encoding="utf-8")


def _assert_contains(actual: str, expected_substring: str, context: str) -> None:
    if expected_substring not in actual:
        raise AssertionError(f"{context} expected '{expected_substring}', got '{actual}'")


def _run_python_sdk_tests(proxy_base_url: str) -> None:
    client = OpenAI(base_url=proxy_base_url, api_key="111")

    response_mapped = client.responses.create(
        model="client-alias-model",
        input="sdk-python-mapped",
    )
    text_mapped = response_mapped.output_text or ""
    _assert_contains(
        text_mapped,
        "mock:gpt-5.3-codex:sdk-python-mapped:effort=high",
        "python responses mapped model",
    )

    response_direct = client.responses.create(
        model="gpt-5.3-codex",
        input="sdk-python-direct",
    )
    text_direct = response_direct.output_text or ""
    _assert_contains(
        text_direct,
        "mock:gpt-5.3-codex:sdk-python-direct:effort=high",
        "python responses direct model",
    )

    response_inline_effort = client.responses.create(
        model="gpt-5.3-codex:medium",
        input="sdk-python-inline-effort",
    )
    text_inline_effort = response_inline_effort.output_text or ""
    _assert_contains(
        text_inline_effort,
        "mock:gpt-5.3-codex:sdk-python-inline-effort:effort=medium",
        "python responses inline effort",
    )

    response_inline_unsupported = client.responses.create(
        model="gpt-4o:high",
        input="sdk-python-inline-unsupported",
    )
    text_inline_unsupported = response_inline_unsupported.output_text or ""
    _assert_contains(
        text_inline_unsupported,
        "mock:gpt-4o:sdk-python-inline-unsupported:effort=none",
        "python responses strip effort for non-codex",
    )

    saw_delta = False
    stream = client.responses.create(
        model="gpt-5.3-codex",
        input="sdk-python-stream",
        stream=True,
    )
    for event in stream:
        event_type = getattr(event, "type", None)
        if event_type == "response.output_text.delta":
            saw_delta = True
            break
    if not saw_delta:
        raise AssertionError("python responses stream did not yield output_text delta")

    chat = client.chat.completions.create(
        model="client-alias-model",
        messages=[{"role": "user", "content": "sdk-python-chat"}],
    )
    chat_text = (chat.choices[0].message.content or "") if chat.choices else ""
    _assert_contains(
        chat_text,
        "mock:gpt-5.3-codex:sdk-python-chat:effort=high",
        "python chat completion",
    )

    chat_inline = client.chat.completions.create(
        model="gpt-5.3-codex:low",
        messages=[{"role": "user", "content": "sdk-python-chat-inline"}],
    )
    chat_inline_text = (chat_inline.choices[0].message.content or "") if chat_inline.choices else ""
    _assert_contains(
        chat_inline_text,
        "mock:gpt-5.3-codex:sdk-python-chat-inline:effort=low",
        "python chat inline effort",
    )

    chat_non_codex = client.chat.completions.create(
        model="gpt-4o:high",
        messages=[{"role": "user", "content": "sdk-python-chat-non-codex"}],
    )
    chat_non_codex_text = (chat_non_codex.choices[0].message.content or "") if chat_non_codex.choices else ""
    _assert_contains(
        chat_non_codex_text,
        "mock:gpt-4o:sdk-python-chat-non-codex:effort=none",
        "python chat strip effort for non-codex",
    )


def _ensure_node_sdk_installed() -> None:
    NODE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    package_json = NODE_TEST_DIR / "package.json"
    if not package_json.exists():
        _run(["npm", "init", "-y"], cwd=NODE_TEST_DIR)
    try:
        _run(["npm", "ls", "openai"], cwd=NODE_TEST_DIR)
    except subprocess.CalledProcessError:
        _run(["npm", "install", "openai"], cwd=NODE_TEST_DIR)


def _run_node_sdk_tests(proxy_base_url: str) -> None:
    script_path = NODE_TEST_DIR / "sdk_test.mjs"
    script_path.write_text(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import OpenAI from "openai";

            const baseURL = process.env.PROXY_BASE_URL;
            const client = new OpenAI({ baseURL, apiKey: "111" });

            const explicitResp = await client.responses.create({
              model: "client-alias-model",
              input: "sdk-node-mapped",
            });
            assert.ok(
              (explicitResp.output_text || "").includes("mock:gpt-5.3-codex:sdk-node-mapped:effort=high"),
              "node responses mapped model failed",
            );

            const directResp = await client.responses.create({
              model: "gpt-5.3-codex",
              input: "sdk-node-direct",
            });
            assert.ok(
              (directResp.output_text || "").includes("mock:gpt-5.3-codex:sdk-node-direct:effort=high"),
              "node responses direct model failed",
            );

            const inlineResp = await client.responses.create({
              model: "gpt-5.3-codex:xhigh",
              input: "sdk-node-inline",
            });
            assert.ok(
              (inlineResp.output_text || "").includes("mock:gpt-5.3-codex:sdk-node-inline:effort=xhigh"),
              "node responses inline effort failed",
            );

            const nonCodexResp = await client.responses.create({
              model: "gpt-4o:high",
              input: "sdk-node-non-codex",
            });
            assert.ok(
              (nonCodexResp.output_text || "").includes("mock:gpt-4o:sdk-node-non-codex:effort=none"),
              "node responses strip effort for non-codex failed",
            );

            let sawDelta = false;
            const stream = await client.responses.create({
              model: "gpt-5.3-codex",
              input: "sdk-node-stream",
              stream: true,
            });
            for await (const event of stream) {
              if (event?.type === "response.output_text.delta") {
                sawDelta = true;
                break;
              }
            }
            assert.ok(sawDelta, "node responses stream did not yield output_text delta");

            const chat = await client.chat.completions.create({
              model: "client-alias-model",
              messages: [{ role: "user", content: "sdk-node-chat" }],
            });
            const content = chat.choices?.[0]?.message?.content || "";
            assert.ok(
              content.includes("mock:gpt-5.3-codex:sdk-node-chat:effort=high"),
              "node chat completion failed",
            );

            const chatInline = await client.chat.completions.create({
              model: "gpt-5.3-codex:medium",
              messages: [{ role: "user", content: "sdk-node-chat-inline" }],
            });
            const chatInlineContent = chatInline.choices?.[0]?.message?.content || "";
            assert.ok(
              chatInlineContent.includes("mock:gpt-5.3-codex:sdk-node-chat-inline:effort=medium"),
              "node chat inline effort failed",
            );

            const chatNonCodex = await client.chat.completions.create({
              model: "gpt-4o:high",
              messages: [{ role: "user", content: "sdk-node-chat-non-codex" }],
            });
            const chatNonCodexContent = chatNonCodex.choices?.[0]?.message?.content || "";
            assert.ok(
              chatNonCodexContent.includes("mock:gpt-4o:sdk-node-chat-non-codex:effort=none"),
              "node chat strip effort for non-codex failed",
            );

            console.log("node sdk tests passed");
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PROXY_BASE_URL"] = proxy_base_url
    result = _run(["node", str(script_path)], cwd=NODE_TEST_DIR, env=env)
    print(result.stdout.strip())


def main() -> int:
    _ensure_node_sdk_installed()

    upstream_port = _find_free_port()
    proxy_port = _find_free_port()

    with tempfile.TemporaryDirectory(prefix="mock-upstream-") as temp_dir:
        temp_path = Path(temp_dir)
        mock_script = temp_path / "mock_upstream.py"
        _write_mock_upstream_script(mock_script)

        mock_process = subprocess.Popen(
            [sys.executable, str(mock_script), str(upstream_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        proxy_env = os.environ.copy()
        proxy_env.update(
            {
                "APP_HOST": "127.0.0.1",
                "APP_PORT": str(proxy_port),
                "UPSTREAM_BASE_URL": f"http://127.0.0.1:{upstream_port}/v1",
                "UPSTREAM_OPENAI_API_KEY": "test-upstream-key",
                "UPSTREAM_ANTIGRAVITY_API_KEY": "test-antigravity-key",
                "DEFAULT_UPSTREAM_MODEL": "gpt-5.3-codex",
                "MODEL_MAP": '{"client-alias-model":"gpt-5.3-codex"}',
                "DEFAULT_REASONING_EFFORT": "high",
                "RAW_IO_LOG_ENABLED": "false",
            }
        )

        proxy_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(proxy_port)],
            cwd=str(REPO_ROOT),
            env=proxy_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            _wait_http_ready(f"http://127.0.0.1:{upstream_port}/v1/responses")
            _wait_http_ready(f"http://127.0.0.1:{proxy_port}/healthz")
            proxy_base_url = f"http://127.0.0.1:{proxy_port}/v1"

            fallback_response = httpx.post(
                f"{proxy_base_url}/responses",
                json={"input": "sdk-http-default"},
                headers={"Authorization": "Bearer 111"},
                timeout=10,
            )
            fallback_response.raise_for_status()
            fallback_json = fallback_response.json()
            fallback_text = _extract_input_text(fallback_json.get("output", []))
            _assert_contains(
                fallback_text,
                "mock:gpt-5.3-codex:sdk-http-default:effort=high",
                "http default model fallback",
            )
            print("http default model fallback passed")

            _run_python_sdk_tests(proxy_base_url)
            print("python sdk tests passed")

            _run_node_sdk_tests(proxy_base_url)
            print("all sdk e2e tests passed")
            return 0
        finally:
            for process in (proxy_process, mock_process):
                if process.poll() is None:
                    process.terminate()
            for process in (proxy_process, mock_process):
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
