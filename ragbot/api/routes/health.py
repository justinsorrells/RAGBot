"""Health routes."""

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ragbot.api.dependencies import get_rag_service
from ragbot.core.service import RAGService

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness response payload."""

    status: str
    documents_indexed: int


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_service: RAGService = Depends(get_rag_service)) -> HealthResponse:
    """Return a liveness response and the number of indexed documents."""

    document_count = await asyncio.to_thread(rag_service.document_count)
    return HealthResponse(status="ok", documents_indexed=document_count)
