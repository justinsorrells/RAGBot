"""Shared test helpers."""

import re

from langchain_core.embeddings import Embeddings

from ragbot.config import Settings
from ragbot.core.indexer import DocumentIndexer, PersistentVectorIndex
from ragbot.core.retriever import DocumentRetriever, RetrievedChunk
from ragbot.core.service import RAGService
from ragbot.db.session import create_engine_from_url, create_session_factory, init_database


class FakeEmbeddings(Embeddings):
    """Deterministic local embeddings used in tests."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 24
        for token in re.findall(r"\w+", text.lower()):
            vector[sum(ord(character) for character in token) % len(vector)] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLMClient:
    """Simple answer generator for API integration tests."""

    def answer_question(self, question: str, sources: list[RetrievedChunk]) -> str:
        combined_context = " ".join(source.text for source in sources)
        return f"Question: {question} | Context: {combined_context[:140]}"


def build_test_service(settings: Settings) -> tuple[RAGService, object]:
    """Build a fully local RAG service for tests."""

    engine = create_engine_from_url(settings.database_url)
    init_database(engine)
    session_factory = create_session_factory(engine)
    vector_index = PersistentVectorIndex(FakeEmbeddings(), settings.faiss_path)
    indexer = DocumentIndexer(vector_index)
    retriever = DocumentRetriever(vector_index)
    service = RAGService(
        session_factory=session_factory,
        indexer=indexer,
        retriever=retriever,
        llm_client=FakeLLMClient(),
        default_top_k=settings.default_top_k,
    )
    return service, engine
