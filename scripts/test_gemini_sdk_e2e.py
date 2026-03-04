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
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_FALLBACK_MODEL = "gemini-3-flash-preview"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http_ready(url: str, timeout_sec: float = 25.0) -> None:
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
        time.sleep(0.3)
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


def _assert_contains(actual: str, expected_substring: str, context: str) -> None:
    if expected_substring not in actual:
        raise AssertionError(f"{context} expected '{expected_substring}', got '{actual}'")


def _write_mock_gemini_script(path: Path) -> None:
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

        PRIMARY_MODEL = "gemini-3.1-pro-preview"
        FALLBACK_MODEL = "gemini-3-flash-preview"
        app = FastAPI()
        retry_once_done = False


        def _error(status_code: int, message: str, status: str, field: str | None = None):
            payload = {"error": {"code": status_code, "message": message, "status": status}}
            if field:
                payload["error"]["details"] = [{"fieldViolations": [{"field": field}]}]
            return JSONResponse(status_code=status_code, content=payload)


        def _extract_user_text(payload: dict[str, Any]) -> str:
            contents = payload.get("contents")
            if not isinstance(contents, list):
                return ""
            chunks: list[str] = []
            for item in contents:
                if not isinstance(item, dict):
                    continue
                parts = item.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return " ".join(chunks).strip()


        def _find_function_response(payload: dict[str, Any]) -> dict[str, Any] | None:
            contents = payload.get("contents")
            if not isinstance(contents, list):
                return None
            for item in contents:
                if not isinstance(item, dict):
                    continue
                parts = item.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    function_response = part.get("functionResponse")
                    if isinstance(function_response, dict):
                        return function_response
            return None


        def _has_tools(payload: dict[str, Any]) -> bool:
            tools = payload.get("tools")
            if not isinstance(tools, list):
                return False
            return len(tools) > 0


        def _response_base() -> dict[str, Any]:
            return {
                "createTime": "2026-02-27T12:00:00Z",
                "responseId": "gemini_mock_resp_1",
                "modelVersion": "mock-1.0",
            }


        @app.get("/v1beta/models")
        async def list_models():
            return {"models": [{"name": f"models/{PRIMARY_MODEL}"}, {"name": f"models/{FALLBACK_MODEL}"}]}


        @app.post("/v1beta/models/{model}:generateContent")
        async def generate_content(model: str, request: Request):
            global retry_once_done
            payload = await request.json()
            if model not in {PRIMARY_MODEL, FALLBACK_MODEL}:
                return _error(404, "Model not found", "NOT_FOUND")

            generation_config = payload.get("generationConfig")
            if isinstance(generation_config, dict):
                max_output_tokens = generation_config.get("maxOutputTokens")
                if isinstance(max_output_tokens, int) and max_output_tokens > 65536:
                    return _error(
                        400,
                        "maxOutputTokens too large",
                        "INVALID_ARGUMENT",
                        "generationConfig.maxOutputTokens",
                    )

            user_text = _extract_user_text(payload)
            if "__force_503_once__" in user_text and not retry_once_done:
                retry_once_done = True
                return _error(503, "No available Gemini accounts: no available accounts", "INTERNAL")
            if "__force_fallback__" in user_text and model == PRIMARY_MODEL:
                return _error(503, "No available Gemini accounts: no available accounts", "INTERNAL")

            function_response = _find_function_response(payload)
            if function_response is not None and not isinstance(function_response.get("name"), str):
                return _error(400, "functionResponse.name is required", "INVALID_ARGUMENT", "contents.parts.functionResponse.name")

            body = _response_base()
            if function_response is not None:
                result = function_response.get("response")
                temp = ""
                if isinstance(result, dict):
                    temp = str(result.get("temperature", ""))
                body["candidates"] = [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": f"tool-result:{temp}"}]},
                    }
                ]
                body["usageMetadata"] = {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 6,
                    "totalTokenCount": 26,
                }
                return JSONResponse(status_code=200, content=body)

            if _has_tools(payload) and "weather" in user_text.lower():
                body["candidates"] = [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "get_weather", "args": {"city": "Boston"}}}
                            ]
                        },
                    }
                ]
                body["usageMetadata"] = {
                    "promptTokenCount": 18,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 22,
                }
                return JSONResponse(status_code=200, content=body)

            text_prefix = "gemini-fallback-mock" if model == FALLBACK_MODEL else "gemini-mock"
            body["candidates"] = [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": f"{text_prefix}:{user_text}"}]},
                }
            ]
            body["usageMetadata"] = {
                "promptTokenCount": 12,
                "candidatesTokenCount": 5,
                "totalTokenCount": 17,
            }
            return JSONResponse(status_code=200, content=body)


        @app.post("/v1beta/models/{model}:streamGenerateContent")
        async def stream_generate_content(model: str, request: Request):
            payload = await request.json()
            if model not in {PRIMARY_MODEL, FALLBACK_MODEL}:
                return _error(404, "Model not found", "NOT_FOUND")

            user_text = _extract_user_text(payload)
            is_tool_call = _has_tools(payload) and "weather" in user_text.lower()

            async def iterator():
                if is_tool_call:
                    chunk = {
                        "createTime": "2026-02-27T12:00:01Z",
                        "responseId": "gemini_stream_tool_1",
                        "candidates": [
                            {
                                "finishReason": "STOP",
                                "content": {
                                    "parts": [
                                        {"functionCall": {"name": "get_weather", "args": {"city": "Boston"}}}
                                    ]
                                },
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 15,
                            "candidatesTokenCount": 4,
                            "totalTokenCount": 19,
                        },
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\\n\\n"
                    return

                chunk1 = {
                    "createTime": "2026-02-27T12:00:01Z",
                    "responseId": "gemini_stream_text_1",
                    "candidates": [{"content": {"parts": [{"text": "stream-mock:"}]}}],
                }
                chunk2 = {
                    "createTime": "2026-02-27T12:00:02Z",
                    "responseId": "gemini_stream_text_1",
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": f"stream-mock:{user_text}"}]},
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 16,
                        "candidatesTokenCount": 6,
                        "totalTokenCount": 22,
                    },
                }
                yield f"data: {json.dumps(chunk1, ensure_ascii=False)}\\n\\n"
                yield f"data: {json.dumps(chunk2, ensure_ascii=False)}\\n\\n"

            return StreamingResponse(iterator(), media_type="text/event-stream")


        if __name__ == "__main__":
            port = int(sys.argv[1])
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
        """
    ).strip()
    path.write_text(source, encoding="utf-8")


def _run_python_sdk_tests(proxy_base_url: str) -> None:
    client = OpenAI(base_url=proxy_base_url, api_key="111")

    response = client.responses.create(
        model=GEMINI_MODEL,
        input="hello",
    )
    text = response.output_text or ""
    _assert_contains(text, "gemini-mock:hello", "python responses text")

    retry_response = client.responses.create(
        model=GEMINI_MODEL,
        input="__force_503_once__ hello",
    )
    retry_text = retry_response.output_text or ""
    _assert_contains(retry_text, "gemini-mock:__force_503_once__ hello", "python responses retry")

    fallback_response = client.responses.create(
        model=GEMINI_MODEL,
        input="__force_fallback__ hello",
    )
    fallback_text = fallback_response.output_text or ""
    _assert_contains(
        fallback_text,
        "gemini-fallback-mock:__force_fallback__ hello",
        "python responses fallback model",
    )

    bounded_response = client.responses.create(
        model=GEMINI_MODEL,
        input="hello with oversized max_output_tokens",
        max_output_tokens=9999999,
    )
    bounded_text = bounded_response.output_text or ""
    _assert_contains(
        bounded_text,
        "gemini-mock:hello with oversized max_output_tokens",
        "python responses max_output_tokens clamp",
    )

    oversized_completion = httpx.post(
        f"{proxy_base_url}/completions",
        headers={"Authorization": "Bearer 111"},
        json={
            "model": GEMINI_MODEL,
            "prompt": "hello oversized max_tokens",
            "max_tokens": 9999999,
            "stream": False,
        },
        timeout=20,
    )
    if oversized_completion.status_code != 200:
        raise AssertionError(
            f"python completions max_tokens clamp expected 200, got {oversized_completion.status_code}"
        )

    stream = client.responses.create(
        model=GEMINI_MODEL,
        input="hello-stream",
        stream=True,
    )
    saw_delta = False
    for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            saw_delta = True
            break
    if not saw_delta:
        raise AssertionError("python responses stream did not yield output_text delta")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather by city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    chat_tool_call = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": "weather for Boston"}],
        tools=tools,
    )
    if not chat_tool_call.choices:
        raise AssertionError("python chat tool call: no choices")
    choice = chat_tool_call.choices[0]
    if choice.finish_reason != "tool_calls":
        raise AssertionError(f"python chat tool call expected finish_reason=tool_calls, got {choice.finish_reason}")
    tool_calls = choice.message.tool_calls or []
    if not tool_calls:
        raise AssertionError("python chat tool call missing tool_calls")

    tool_call = tool_calls[0]
    tool_stream = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": "weather for Boston"}],
        tools=tools,
        stream=True,
    )
    saw_stream_tool_call = False
    for chunk in tool_stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        if getattr(delta, "tool_calls", None):
            saw_stream_tool_call = True
            break
    if not saw_stream_tool_call:
        raise AssertionError("python chat stream tool call delta missing")

    fallback_chat = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": "__force_fallback__ say hi"}],
        max_tokens=64,
    )
    fallback_chat_text = (fallback_chat.choices[0].message.content or "") if fallback_chat.choices else ""
    _assert_contains(
        fallback_chat_text,
        "gemini-fallback-mock:__force_fallback__ say hi",
        "python chat fallback model",
    )

    followup = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "user", "content": "weather for Boston"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": '{"temperature":"10C"}',
            },
        ],
        tools=tools,
    )
    followup_text = followup.choices[0].message.content or ""
    _assert_contains(followup_text, "tool-result:10C", "python chat functionResponse")

    invalid_model_response = httpx.post(
        f"{proxy_base_url}/responses",
        headers={"Authorization": "Bearer 111"},
        json={"model": "gemini-not-found", "input": "hello"},
        timeout=20,
    )
    if invalid_model_response.status_code != 404:
        raise AssertionError(f"invalid model status expected 404, got {invalid_model_response.status_code}")
    invalid_model_body = invalid_model_response.json()
    error_obj = invalid_model_body.get("error", {})
    if error_obj.get("type") != "invalid_request_error":
        raise AssertionError(f"invalid model error type expected invalid_request_error, got {error_obj.get('type')}")

    malformed_payload_response = httpx.post(
        f"{proxy_base_url}/responses",
        headers={"Authorization": "Bearer 111"},
        json={
            "model": GEMINI_MODEL,
            "input": [
                {"type": "function_call", "name": "bad_tool", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "missing", "output": '{"ok":true}'},
            ],
        },
        timeout=20,
    )
    if malformed_payload_response.status_code != 400:
        raise AssertionError(
            f"malformed function_call_output expected 400, got {malformed_payload_response.status_code}"
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
    script_path = NODE_TEST_DIR / "gemini_sdk_test.mjs"
    script_path.write_text(
        textwrap.dedent(
            """
            import assert from "node:assert/strict";
            import OpenAI from "openai";

            const baseURL = process.env.PROXY_BASE_URL;
            const model = process.env.GEMINI_MODEL;
            const client = new OpenAI({ baseURL, apiKey: "111" });

            const response = await client.responses.create({
              model,
              input: "node-hello",
            });
            assert.ok(
              (response.output_text || "").includes("gemini-mock:node-hello"),
              "node responses text failed",
            );

            const chatStream = await client.chat.completions.create({
              model,
              messages: [{ role: "user", content: "stream from node" }],
              stream: true,
            });
            let sawDelta = false;
            for await (const event of chatStream) {
              const delta = event?.choices?.[0]?.delta?.content;
              if (typeof delta === "string" && delta.length > 0) {
                sawDelta = true;
                break;
              }
            }
            assert.ok(sawDelta, "node chat stream did not yield content delta");
            console.log("node gemini sdk tests passed");
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PROXY_BASE_URL"] = proxy_base_url
    env["GEMINI_MODEL"] = GEMINI_MODEL
    result = _run(["node", str(script_path)], cwd=NODE_TEST_DIR, env=env)
    print(result.stdout.strip())


def main() -> int:
    _ensure_node_sdk_installed()
    gemini_port = _find_free_port()
    openai_mock_port = _find_free_port()
    proxy_port = _find_free_port()

    with tempfile.TemporaryDirectory(prefix="mock-gemini-upstream-") as temp_dir:
        temp_path = Path(temp_dir)
        gemini_script = temp_path / "mock_gemini.py"
        gemini_script.write_text("", encoding="utf-8")
        _write_mock_gemini_script(gemini_script)

        openai_mock_script = temp_path / "mock_openai.py"
        openai_mock_script.write_text(
            textwrap.dedent(
                """
                from __future__ import annotations

                import sys
                import time

                import uvicorn
                from fastapi import FastAPI, Request
                from fastapi.responses import JSONResponse

                app = FastAPI()


                @app.post("/v1/responses")
                async def responses(request: Request):
                    payload = await request.json()
                    model = payload.get("model", "openai-default")
                    text = "openai-mock"
                    return JSONResponse(
                        status_code=200,
                        content={
                            "id": f"resp_openai_{int(time.time())}",
                            "object": "response",
                            "created_at": int(time.time()),
                            "status": "completed",
                            "model": model,
                            "output": [
                                {
                                    "id": "msg_openai",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                                }
                            ],
                            "output_text": text,
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        },
                    )


                if __name__ == "__main__":
                    port = int(sys.argv[1])
                    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        gemini_process = subprocess.Popen(
            [sys.executable, str(gemini_script), str(gemini_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        openai_mock_process = subprocess.Popen(
            [sys.executable, str(openai_mock_script), str(openai_mock_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        proxy_env = os.environ.copy()
        proxy_env.update(
            {
                "APP_HOST": "127.0.0.1",
                "APP_PORT": str(proxy_port),
                "UPSTREAM_BASE_URL": f"http://127.0.0.1:{openai_mock_port}/v1",
                "UPSTREAM_API_KEY": "test-openai-key",
                "UPSTREAM_GEMINI_BASE_URL": f"http://127.0.0.1:{gemini_port}/v1beta",
                "UPSTREAM_GEMINI_API_KEY": "test-gemini-key",
                "GEMINI_MIN_REQUEST_INTERVAL_SECONDS": "0",
                "GEMINI_FALLBACK_MODEL": GEMINI_FALLBACK_MODEL,
                "DEFAULT_UPSTREAM_MODEL": "gpt-5.3-codex",
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
            _wait_http_ready(f"http://127.0.0.1:{gemini_port}/v1beta/models")
            _wait_http_ready(f"http://127.0.0.1:{openai_mock_port}/v1/responses")
            _wait_http_ready(f"http://127.0.0.1:{proxy_port}/healthz")

            proxy_base = f"http://127.0.0.1:{proxy_port}/v1"
            _run_python_sdk_tests(proxy_base)
            print("python gemini sdk tests passed")
            _run_node_sdk_tests(proxy_base)
            print("all gemini e2e tests passed")
            return 0
        finally:
            for process in (proxy_process, gemini_process, openai_mock_process):
                if process.poll() is None:
                    process.terminate()
            for process in (proxy_process, gemini_process, openai_mock_process):
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
