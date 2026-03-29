"""Document upload and listing routes."""

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict

from ragbot.api.dependencies import get_rag_service
from ragbot.core.service import RAGService

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    """API response model for indexed document metadata."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    chunk_count: int
    upload_time: datetime


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentResponse:
    """Upload a text or PDF document, chunk it, and store it in FAISS and SQLite."""

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A filename is required.")

    content = await file.read()
    try:
        record = await asyncio.to_thread(rag_service.add_document, file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return DocumentResponse.model_validate(record)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(rag_service: RAGService = Depends(get_rag_service)) -> list[DocumentResponse]:
    """List all indexed documents and their stored metadata."""

    records = await asyncio.to_thread(rag_service.list_documents)
    return [DocumentResponse.model_validate(record) for record in records]
