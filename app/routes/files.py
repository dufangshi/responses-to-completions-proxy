from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from app.services.file_store import LocalFileStore

router = APIRouter()


def _file_store(request: Request) -> LocalFileStore:
    return request.app.state.file_store


@router.post("/v1/files")
@router.post("/files")
async def create_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form(...),
) -> JSONResponse:
    store = _file_store(request)
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must not be empty.",
        )
    record = store.create_file(
        filename=file.filename or "upload.bin",
        purpose=purpose,
        content=content,
        content_type=file.content_type,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=record.to_openai_dict())


@router.get("/v1/files")
@router.get("/files")
async def list_files(request: Request) -> JSONResponse:
    store = _file_store(request)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "object": "list",
            "data": [record.to_openai_dict() for record in store.list_files()],
            "has_more": False,
        },
    )


@router.get("/v1/files/{file_id}")
@router.get("/files/{file_id}")
async def retrieve_file(file_id: str, request: Request) -> JSONResponse:
    store = _file_store(request)
    record = store.get_file(file_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return JSONResponse(status_code=status.HTTP_200_OK, content=record.to_openai_dict())


@router.get("/v1/files/{file_id}/content")
@router.get("/files/{file_id}/content")
async def retrieve_file_content(file_id: str, request: Request) -> Response:
    store = _file_store(request)
    record = store.get_file(file_id)
    content = store.read_bytes(file_id)
    if record is None or content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")
    return Response(
        content=content,
        media_type=record.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{record.filename}"',
        },
    )


@router.delete("/v1/files/{file_id}")
@router.delete("/files/{file_id}")
async def delete_file(file_id: str, request: Request) -> JSONResponse:
    store = _file_store(request)
    deleted = store.delete_file(file_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"id": file_id, "object": "file", "deleted": deleted},
    )
