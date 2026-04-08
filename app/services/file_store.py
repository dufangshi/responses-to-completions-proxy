from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_INVALID_FILENAME_CHARS = re.compile(r'[\\/\0\r\n\t"]+')
_COLLAPSE_SPACES = re.compile(r"\s+")


def normalize_user_supplied_filename(
    filename: str | None,
    fallback: str = "upload.bin",
) -> str:
    fallback_name = fallback.strip() or "upload.bin"
    if not isinstance(filename, str):
        return fallback_name

    sanitized = _INVALID_FILENAME_CHARS.sub(" ", filename.strip())
    sanitized = _COLLAPSE_SPACES.sub(" ", sanitized).strip(" .")
    return sanitized or fallback_name


def normalize_input_file_reference_for_cache(part: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "type": "input_file",
        "filename": normalize_user_supplied_filename(part.get("filename"), fallback="upload.bin"),
    }

    file_id = part.get("file_id")
    if isinstance(file_id, str) and file_id.strip():
        normalized["file_id"] = file_id.strip()

    file_url = part.get("file_url")
    if isinstance(file_url, str) and file_url.strip():
        normalized["file_url"] = file_url.strip()

    file_data = part.get("file_data")
    if isinstance(file_data, str) and file_data.strip():
        normalized["file_digest"] = hashlib.sha256(file_data.strip().encode("utf-8")).hexdigest()[:24]

    return normalized


@dataclass(slots=True)
class StoredFileRecord:
    id: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    content_type: str
    status: str = "processed"
    object: str = "file"
    status_details: str | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "bytes": self.bytes,
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": self.purpose,
            "status": self.status,
            "status_details": self.status_details,
        }


class LocalFileStore:
    def __init__(self, root_dir: str | Path):
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def create_file(
        self,
        *,
        filename: str,
        purpose: str,
        content: bytes,
        content_type: str | None = None,
    ) -> StoredFileRecord:
        file_id = f"file-{uuid.uuid4().hex}"
        created_at = int(time.time())
        resolved_filename = normalize_user_supplied_filename(filename, fallback="upload.bin")
        resolved_content_type = (
            content_type.strip() if isinstance(content_type, str) and content_type.strip() else "application/octet-stream"
        )
        record = StoredFileRecord(
            id=file_id,
            bytes=len(content),
            created_at=created_at,
            filename=resolved_filename,
            purpose=purpose.strip() or "user_data",
            content_type=resolved_content_type,
        )
        self._metadata_path(file_id).write_text(
            json.dumps(
                {
                    "id": record.id,
                    "bytes": record.bytes,
                    "created_at": record.created_at,
                    "filename": record.filename,
                    "purpose": record.purpose,
                    "content_type": record.content_type,
                    "status": record.status,
                    "object": record.object,
                    "status_details": record.status_details,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._blob_path(file_id).write_bytes(content)
        return record

    def get_file(self, file_id: str) -> StoredFileRecord | None:
        metadata_path = self._metadata_path(file_id)
        if not metadata_path.exists():
            return None
        try:
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return StoredFileRecord(
                id=str(raw["id"]),
                bytes=int(raw["bytes"]),
                created_at=int(raw["created_at"]),
                filename=str(raw["filename"]),
                purpose=str(raw["purpose"]),
                content_type=str(raw.get("content_type") or "application/octet-stream"),
                status=str(raw.get("status") or "processed"),
                object=str(raw.get("object") or "file"),
                status_details=raw.get("status_details"),
            )
        except Exception:
            return None

    def list_files(self) -> list[StoredFileRecord]:
        records: list[StoredFileRecord] = []
        for metadata_path in sorted(self._root_dir.glob("*.json")):
            record = self.get_file(metadata_path.stem)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records

    def delete_file(self, file_id: str) -> bool:
        metadata_path = self._metadata_path(file_id)
        blob_path = self._blob_path(file_id)
        existed = False
        if metadata_path.exists():
            metadata_path.unlink()
            existed = True
        if blob_path.exists():
            blob_path.unlink()
            existed = True
        return existed

    def read_bytes(self, file_id: str) -> bytes | None:
        blob_path = self._blob_path(file_id)
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()

    def build_data_url(self, file_id: str) -> tuple[str, str, str] | None:
        record = self.get_file(file_id)
        if record is None:
            return None
        content = self.read_bytes(file_id)
        if content is None:
            return None
        base64_data = base64.b64encode(content).decode("ascii")
        return record.filename, record.content_type, f"data:{record.content_type};base64,{base64_data}"

    def _metadata_path(self, file_id: str) -> Path:
        return self._root_dir / f"{file_id}.json"

    def _blob_path(self, file_id: str) -> Path:
        return self._root_dir / f"{file_id}.bin"


def resolve_openai_payload_file_ids(payload: dict[str, Any], store: LocalFileStore) -> dict[str, Any]:
    return _resolve_openai_value(payload, store)


def resolve_native_message_file_ids(payload: dict[str, Any], store: LocalFileStore) -> dict[str, Any]:
    resolved = _resolve_native_value(payload, store)
    if isinstance(resolved, dict):
        return resolved
    return dict(payload)


def _resolve_openai_value(value: Any, store: LocalFileStore) -> Any:
    if isinstance(value, list):
        return [_resolve_openai_value(item, store) for item in value]

    if not isinstance(value, dict):
        return value

    part_type = value.get("type")
    if part_type == "input_file":
        return _resolve_openai_input_file_part(value, store)
    if part_type == "file":
        file_value = value.get("file")
        if isinstance(file_value, dict):
            resolved_file = _resolve_openai_input_file_part(file_value, store)
            return {"type": "input_file", **resolved_file}

    return {key: _resolve_openai_value(item, store) for key, item in value.items()}


def _resolve_openai_input_file_part(part: dict[str, Any], store: LocalFileStore) -> dict[str, Any]:
    resolved = {key: _resolve_openai_value(value, store) for key, value in part.items()}
    file_id = resolved.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        if "filename" in resolved:
            resolved["filename"] = normalize_user_supplied_filename(
                resolved.get("filename"),
                fallback="upload.bin",
            )
        return resolved

    data_url = store.build_data_url(file_id.strip())
    if data_url is None:
        return resolved

    filename, _, file_data = data_url
    rebuilt = dict(resolved)
    rebuilt.pop("file_id", None)
    rebuilt["filename"] = normalize_user_supplied_filename(
        rebuilt.get("filename") or filename,
        fallback=filename,
    )
    rebuilt["file_data"] = file_data
    return rebuilt


def _resolve_native_value(value: Any, store: LocalFileStore) -> Any:
    if isinstance(value, list):
        return [_resolve_native_value(item, store) for item in value]

    if not isinstance(value, dict):
        return value

    block_type = value.get("type")
    if block_type == "document":
        return _resolve_native_document_block(value, store)
    if block_type == "input_file":
        return _resolve_native_input_file_block(value, store)

    return {key: _resolve_native_value(item, store) for key, item in value.items()}


def _resolve_native_document_block(block: dict[str, Any], store: LocalFileStore) -> dict[str, Any]:
    resolved = {key: _resolve_native_value(value, store) for key, value in block.items()}
    source = resolved.get("source")
    if not isinstance(source, dict):
        return resolved
    if source.get("type") != "file":
        return resolved

    file_id = source.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        return resolved

    data_url = store.build_data_url(file_id.strip())
    if data_url is None:
        return resolved

    filename, content_type, file_data = data_url
    _, _, base64_data = file_data.partition(",")
    rebuilt = dict(resolved)
    rebuilt["title"] = normalize_user_supplied_filename(
        rebuilt.get("title") or filename,
        fallback=filename,
    )
    rebuilt["source"] = {
        "type": "base64",
        "media_type": content_type,
        "data": base64_data,
    }
    return rebuilt


def _resolve_native_input_file_block(block: dict[str, Any], store: LocalFileStore) -> dict[str, Any]:
    resolved = {key: _resolve_native_value(value, store) for key, value in block.items()}
    file_id = resolved.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        return resolved

    data_url = store.build_data_url(file_id.strip())
    if data_url is None:
        return resolved

    filename, content_type, file_data = data_url
    _, _, base64_data = file_data.partition(",")
    return {
        "type": "document",
        "title": normalize_user_supplied_filename(
            resolved.get("filename") or filename,
            fallback=filename,
        ),
        "source": {
            "type": "base64",
            "media_type": content_type,
            "data": base64_data,
        },
    }
