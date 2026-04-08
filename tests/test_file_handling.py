from __future__ import annotations

import tempfile
import unittest

from app.routes.chat_completions import _normalize_chat_session_input_items
from app.routes.messages import _normalize_responses_session_input_items
from app.services.file_store import LocalFileStore, normalize_user_supplied_filename
from app.services.transformers import _build_input_file_part


class FileHandlingTests(unittest.TestCase):
    def test_normalize_user_supplied_filename_sanitizes_control_chars(self) -> None:
        normalized = normalize_user_supplied_filename(' ../weird\tname"\n.pdf ', fallback="upload.bin")
        self.assertEqual(normalized, "weird name .pdf")

    def test_local_file_store_create_file_uses_normalized_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalFileStore(temp_dir)
            record = store.create_file(
                filename=' subdir/notes\t.txt ',
                purpose="user_data",
                content=b"hello",
                content_type="text/plain",
            )

        self.assertEqual(record.filename, "subdir notes .txt")

    def test_build_input_file_part_normalizes_filename(self) -> None:
        part = _build_input_file_part(
            {
                "filename": ' folder/report\tfinal".txt ',
                "file_id": "file-123",
            }
        )

        self.assertEqual(part["filename"], "folder report final .txt")
        self.assertEqual(part["file_id"], "file-123")

    def test_chat_session_normalization_replaces_file_data_with_digest(self) -> None:
        normalized = _normalize_chat_session_input_items(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {
                            "type": "input_file",
                            "filename": ' docs/spec\tv1".pdf ',
                            "file_data": "data:application/pdf;base64,QUJDREVGRw==",
                        },
                    ],
                }
            ]
        )

        file_part = normalized[0]["content"][1]
        self.assertEqual(file_part["filename"], "docs spec v1 .pdf")
        self.assertIn("file_digest", file_part)
        self.assertNotIn("file_data", file_part)

    def test_responses_session_normalization_replaces_file_data_with_digest(self) -> None:
        normalized = _normalize_responses_session_input_items(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": ' docs/notes\tv2".txt ',
                            "file_data": "data:text/plain;base64,SGVsbG8=",
                        }
                    ],
                }
            ]
        )

        file_part = normalized[0]["content"][0]
        self.assertEqual(file_part["filename"], "docs notes v2 .txt")
        self.assertIn("file_digest", file_part)
        self.assertNotIn("file_data", file_part)


if __name__ == "__main__":
    unittest.main()
