"""Application service layer."""

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ragbot.core.indexer import DocumentIndexer, extract_text_from_bytes
from ragbot.core.retriever import DocumentRetriever, RetrievedChunk
from ragbot.db.models import DocumentRecord


class AnswerGenerator(Protocol):
    """Protocol for answer generation implementations."""

    def answer_question(self, question: str, sources: list[RetrievedChunk]) -> str:
        """Generate an answer from retrieved source chunks."""


@dataclass(slots=True)
class ChatResult:
    """Structured chat response returned by the service layer."""

    answer: str
    sources: list[RetrievedChunk]


class RAGService:
    """Coordinate document ingestion, retrieval, and answer generation."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        indexer: DocumentIndexer,
        retriever: DocumentRetriever,
        llm_client: AnswerGenerator,
        default_top_k: int = 4,
    ) -> None:
        self.session_factory = session_factory
        self.indexer = indexer
        self.retriever = retriever
        self.llm_client = llm_client
        self.default_top_k = default_top_k

    def add_document(self, filename: str, content: bytes) -> DocumentRecord:
        """Extract, chunk, index, and persist metadata for an uploaded document."""

        text = extract_text_from_bytes(filename, content)
        document_id = str(uuid4())
        chunks = self.indexer.index_document(text=text, filename=filename, document_id=document_id)

        with self.session_factory() as session:
            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                chunk_count=len(chunks),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_documents(self) -> list[DocumentRecord]:
        """Return all indexed documents, newest first."""

        with self.session_factory() as session:
            statement = select(DocumentRecord).order_by(DocumentRecord.upload_time.desc())
            return list(session.scalars(statement).all())

    def answer_question(self, question: str, top_k: int | None = None) -> ChatResult:
        """Retrieve relevant chunks and generate a grounded answer."""

        effective_top_k = top_k or self.default_top_k
        sources = self.retriever.search(question, limit=effective_top_k)
        if not sources:
            return ChatResult(
                answer="I couldn't find relevant information in the indexed documents.",
                sources=[],
            )
        answer = self.llm_client.answer_question(question, sources)
        return ChatResult(answer=answer, sources=sources)

    def document_count(self) -> int:
        """Return the number of indexed documents stored in SQLite."""

        with self.session_factory() as session:
            count = session.scalar(select(func.count(DocumentRecord.id)))
            return int(count or 0)
