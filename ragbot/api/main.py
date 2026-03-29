"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine

from ragbot.api.routes.chat import router as chat_router
from ragbot.api.routes.documents import router as documents_router
from ragbot.api.routes.health import router as health_router
from ragbot.config import Settings, get_settings
from ragbot.core.indexer import DocumentIndexer, PersistentVectorIndex
from ragbot.core.llm import create_embeddings, create_llm_client
from ragbot.core.retriever import DocumentRetriever
from ragbot.core.service import RAGService
from ragbot.db.session import create_engine_from_url, create_session_factory, init_database


def build_rag_service(settings: Settings) -> tuple[RAGService, Engine]:
    """Create the runtime service graph for the application."""

    engine = create_engine_from_url(settings.database_url)
    init_database(engine)
    session_factory = create_session_factory(engine)
    vector_index = PersistentVectorIndex(create_embeddings(settings), settings.faiss_path)
    indexer = DocumentIndexer(vector_index)
    retriever = DocumentRetriever(vector_index)
    llm_client = create_llm_client(settings)
    service = RAGService(
        session_factory=session_factory,
        indexer=indexer,
        retriever=retriever,
        llm_client=llm_client,
        default_top_k=settings.default_top_k,
    )
    return service, engine


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    resolved_settings = settings or get_settings()
    engine_holder: dict[str, Engine | None] = {"engine": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        if service is not None:
            app.state.rag_service = service
            yield
            return

        rag_service, engine = build_rag_service(resolved_settings)
        engine_holder["engine"] = engine
        app.state.rag_service = rag_service
        yield

        if engine_holder["engine"] is not None:
            engine_holder["engine"].dispose()

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)
    return app


app = create_app()
