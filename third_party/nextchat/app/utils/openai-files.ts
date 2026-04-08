import { getClientConfig } from "@/app/config/client";
import {
  ACCESS_CODE_PREFIX,
  ApiPath,
  OpenaiPath,
} from "@/app/constant";
import { UploadedFileRef } from "@/app/client/api";
import { useAccessStore } from "@/app/store";

const PDF_MIME_TYPE = "application/pdf";
const TEXT_MIME_TYPE = "text/plain";
const PDF_EXTENSION = ".pdf";
const TEXT_EXTENSION = ".txt";
const INVALID_FILENAME_CHARS = /[\\/\0\r\n\t"]/g;
const COLLAPSE_SPACES = /\s+/g;
const LEADING_DOTS = /^\.+/;
const TRAILING_DOTS_AND_SPACES = /[.\s]+$/;
const INVALID_EXTENSION_CHARS = /[^.a-zA-Z0-9_-]/g;
const MAX_BASENAME_LENGTH = 64;

export function isSupportedChatAttachmentFile(file: File) {
  const lowerName = file.name.toLowerCase();
  return (
    file.type === PDF_MIME_TYPE ||
    file.type === TEXT_MIME_TYPE ||
    lowerName.endsWith(PDF_EXTENSION) ||
    lowerName.endsWith(TEXT_EXTENSION)
  );
}

export function normalizeAttachmentFilename(rawFilename: string, file: Blob) {
  const fallbackName =
    file.type === PDF_MIME_TYPE
      ? `document${PDF_EXTENSION}`
      : file.type === TEXT_MIME_TYPE
        ? `document${TEXT_EXTENSION}`
        : "upload.bin";

  const sanitized = rawFilename
    .trim()
    .replace(INVALID_FILENAME_CHARS, " ")
    .replace(COLLAPSE_SPACES, " ")
    .trim();

  const resolvedName = sanitized || fallbackName;
  const fallbackExtension = fallbackName.includes(".")
    ? fallbackName.slice(fallbackName.lastIndexOf("."))
    : "";
  const lastDotIndex = resolvedName.lastIndexOf(".");
  const rawBaseName =
    lastDotIndex > 0 ? resolvedName.slice(0, lastDotIndex) : resolvedName;
  const rawExtension = lastDotIndex > 0 ? resolvedName.slice(lastDotIndex) : "";

  const baseName =
    rawBaseName
      .replace(LEADING_DOTS, "")
      .replace(TRAILING_DOTS_AND_SPACES, "")
      .trim()
      .slice(0, MAX_BASENAME_LENGTH) || "document";
  const extension =
    rawExtension.replace(INVALID_EXTENSION_CHARS, "").trim() ||
    fallbackExtension;
  const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  return `${baseName}-${uniqueSuffix}${extension}`;
}

function getBearerToken(apiKey: string) {
  const trimmed = apiKey.trim();
  if (!trimmed) {
    return "";
  }
  return `Bearer ${trimmed}`;
}

function getUploadHeaders() {
  const accessStore = useAccessStore.getState();
  const headers = new Headers();
  const clientConfig = getClientConfig();

  if (!(clientConfig?.isApp)) {
    headers.set("Accept", "application/json");
  }

  const bearerToken = getBearerToken(accessStore.openaiApiKey);
  if (bearerToken) {
    headers.set("Authorization", bearerToken);
  } else if (
    accessStore.enabledAccessControl() &&
    accessStore.accessCode.trim().length > 0
  ) {
    headers.set(
      "Authorization",
      getBearerToken(ACCESS_CODE_PREFIX + accessStore.accessCode),
    );
  }

  return headers;
}

function createUploadErrorMessage(detail: unknown, filename: string) {
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }
  if (typeof detail === "object" && detail && "detail" in detail) {
    const nestedDetail = (detail as { detail?: unknown }).detail;
    if (typeof nestedDetail === "string" && nestedDetail.trim()) {
      return nestedDetail.trim();
    }
  }
  return `failed to upload ${filename}`;
}

export async function uploadOpenAIFile(
  file: File,
  purpose: string = "user_data",
): Promise<UploadedFileRef> {
  const normalizedFilename = normalizeAttachmentFilename(file.name, file);
  const body = new FormData();
  body.append("file", file, normalizedFilename);
  body.append("purpose", purpose);

  const response = await fetch(`${ApiPath.OpenAI}/${OpenaiPath.FilesPath}`, {
    method: "POST",
    body,
    headers: getUploadHeaders(),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(createUploadErrorMessage(payload, normalizedFilename));
  }

  const fileId =
    payload && typeof payload.id === "string" ? payload.id.trim() : "";
  if (!fileId) {
    throw new Error(`missing file id for ${normalizedFilename}`);
  }

  return {
    fileId,
    filename:
      payload && typeof payload.filename === "string" && payload.filename.trim()
        ? payload.filename.trim()
        : normalizedFilename,
    contentType: file.type || undefined,
    bytes: file.size,
  };
}

export function deleteOpenAIFile(fileId: string) {
  return fetch(`${ApiPath.OpenAI}/${OpenaiPath.FilesPath}/${fileId}`, {
    method: "DELETE",
    headers: getUploadHeaders(),
  });
}
