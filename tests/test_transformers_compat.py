from __future__ import annotations

import unittest

from app.models.legacy_chat_completions import LegacyChatCompletionRequest
from app.services.transformers import UnsupportedParameterError, build_chat_responses_payload


class TransformersCompatibilityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
