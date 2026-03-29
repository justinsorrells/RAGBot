"""Chat routes."""

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ragbot.api.dependencies import get_rag_service
from ragbot.core.service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Incoming user question payload."""

    question: str = Field(min_length=1, description="The question to ask about the indexed documents.")
    top_k: int | None = Field(default=None, ge=1, le=10, description="Override for retrieved chunk count.")


class SourceResponse(BaseModel):
    """Serialized source chunk returned with each answer."""

    filename: str
    chunk_text: str
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    """Chat answer payload."""

    answer: str
    sources: list[SourceResponse]


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, rag_service: RAGService = Depends(get_rag_service)) -> ChatResponse:
    """Answer a question by retrieving similar chunks and prompting the configured LLM."""

    result = await asyncio.to_thread(rag_service.answer_question, payload.question, payload.top_k)
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                filename=source.filename,
                chunk_text=source.text,
                chunk_index=source.chunk_index,
            )
            for source in result.sources
        ],
    )
