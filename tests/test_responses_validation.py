from __future__ import annotations

import unittest

from app.models.legacy_chat_completions import LegacyChatCompletionRequest
from app.services.responses_client import UpstreamAPIError, _collect_completed_response_from_sse
from app.services.responses_payload import ResponsesValidationError, normalize_responses_input
from app.services.transformers import UnsupportedParameterError, build_chat_responses_payload


class ResponsesPayloadValidationTests(unittest.TestCase):
    def test_normalize_responses_input_rejects_overlong_input_name(self) -> None:
        payload = {
            "input": [
                {
                    "role": "user",
                    "name": "x" * 129,
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ]
        }

        with self.assertRaises(ResponsesValidationError) as exc_info:
            normalize_responses_input(payload)

        exc = exc_info.exception
        self.assertEqual(exc.param, "input[0].name")
        self.assertEqual(exc.code, "string_above_max_length")

    def test_build_chat_responses_payload_rejects_overlong_message_name(self) -> None:
        request = LegacyChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "name": "x" * 129,
                    "content": "hello",
                }
            ]
        )

        with self.assertRaises(UnsupportedParameterError) as exc_info:
            build_chat_responses_payload(request, target_model="gpt-5.3-codex")

        self.assertIn("input[0].name must be at most 128 characters.", str(exc_info.exception))

    def test_build_chat_responses_payload_ignores_neutral_unsupported_chat_fields(self) -> None:
        request = LegacyChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            frequency_penalty=0,
            presence_penalty=0,
            logit_bias={},
            response_format={"type": "text"},
        )

        payload = build_chat_responses_payload(request, target_model="gpt-5.4")

        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertNotIn("frequency_penalty", payload)
        self.assertNotIn("presence_penalty", payload)

    def test_build_chat_responses_payload_rejects_non_default_frequency_penalty(self) -> None:
        request = LegacyChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            frequency_penalty=0.2,
        )

        with self.assertRaises(UnsupportedParameterError) as exc_info:
            build_chat_responses_payload(request, target_model="gpt-5.4")

        self.assertIn("frequency_penalty", str(exc_info.exception))


class ResponsesClientStreamErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_completed_response_from_sse_preserves_invalid_request_status(self) -> None:
        async def lines():
            yield 'event: response.failed'
            yield (
                'data: {"type":"response.failed","response":{"status":"failed","error":'
                '{"message":"Invalid \\"input[2].name\\"","type":"invalid_request_error",'
                '"param":"input[2].name","code":"string_above_max_length"}}}'
            )
            yield ""
            yield "data: [DONE]"
            yield ""

        with self.assertRaises(UpstreamAPIError) as exc_info:
            await _collect_completed_response_from_sse(lines())

        exc = exc_info.exception
        self.assertEqual(exc.status_code, 400)
        self.assertEqual(exc.payload["error"]["param"], "input[2].name")
        self.assertEqual(exc.payload["error"]["code"], "string_above_max_length")


if __name__ == "__main__":
    unittest.main()
