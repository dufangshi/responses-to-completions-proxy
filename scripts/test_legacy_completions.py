#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import requests


def main() -> int:
    base_url = os.getenv("PROXY_BASE_URL", "http://127.0.0.1:18010")
    api_key = os.getenv("PROXY_API_KEY", "dummy-local-key")
    model = os.getenv("LEGACY_MODEL", "gpt-5.3-codex")

    payload = {
        "model": model,
        "prompt": "Write one short sentence about FastAPI compatibility proxies.",
        "max_tokens": 80,
        "temperature": 0.2,
        "n": 1,
        "stream": False,
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/v1/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    print(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        print(response.text)
        return 1

    print(json.dumps(body, ensure_ascii=False, indent=2))
    return 0 if response.ok else 1


if __name__ == "__main__":
    sys.exit(main())
