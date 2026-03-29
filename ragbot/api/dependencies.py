"""FastAPI dependency helpers."""

from fastapi import Request

from ragbot.core.service import RAGService


def get_rag_service(request: Request) -> RAGService:
    """Return the shared application service container."""

    return request.app.state.rag_service
