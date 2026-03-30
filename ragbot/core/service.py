"""Application service layer."""

from dataclasses import dataclass
from pathlib import Path
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
        document_store_path: str | Path,
        default_top_k: int = 4,
    ) -> None:
        self.session_factory = session_factory
        self.indexer = indexer
        self.retriever = retriever
        self.llm_client = llm_client
        self.document_store_path = Path(document_store_path)
        self.document_store_path.mkdir(parents=True, exist_ok=True)
        self.default_top_k = default_top_k

    def stored_document_path(self, document_id: str, filename: str) -> Path:
        """Return the on-disk location used for a stored source document."""

        return self.document_store_path / f"{document_id}{Path(filename).suffix.lower()}"

    def _load_record_chunks(self, record: DocumentRecord):
        source_path = self.stored_document_path(record.document_id, record.filename)
        if not source_path.exists():
            raise ValueError(f"Stored source file for '{record.filename}' is missing.")

        text = extract_text_from_bytes(record.filename, source_path.read_bytes())
        chunks = self.indexer.split_text(text=text, filename=record.filename, document_id=record.document_id)
        if not chunks:
            raise ValueError(f"Stored source file for '{record.filename}' could not be chunked.")
        return chunks

    def _rebuild_index_for_records(self, records: list[DocumentRecord]) -> dict[str, int]:
        all_chunks = []
        chunk_counts: dict[str, int] = {}
        for record in records:
            chunks = self._load_record_chunks(record)
            chunk_counts[record.document_id] = len(chunks)
            all_chunks.extend(chunks)
        self.indexer.vector_index.rebuild(all_chunks)
        return chunk_counts

    def add_document(self, filename: str, content: bytes) -> DocumentRecord:
        """Extract, chunk, index, and persist metadata for an uploaded document."""

        text = extract_text_from_bytes(filename, content)
        document_id = str(uuid4())
        source_path = self.stored_document_path(document_id, filename)
        source_path.write_bytes(content)

        try:
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
        except Exception:
            if source_path.exists():
                source_path.unlink()
            with self.session_factory() as session:
                records = list(session.scalars(select(DocumentRecord)).all())
            self._rebuild_index_for_records(records)
            raise

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

    def delete_document(self, document_id: str) -> bool:
        """Delete a document, remove its stored file, and rebuild the vector index."""

        with self.session_factory() as session:
            target = session.scalar(
                select(DocumentRecord).where(DocumentRecord.document_id == document_id)
            )
            if target is None:
                return False

            source_path = self.stored_document_path(target.document_id, target.filename)
            remaining_records = list(
                session.scalars(
                    select(DocumentRecord).where(DocumentRecord.document_id != document_id)
                ).all()
            )
            self._rebuild_index_for_records(remaining_records)
            try:
                session.delete(target)
                session.commit()
            except Exception:
                session.rollback()
                all_records = [target, *remaining_records]
                self._rebuild_index_for_records(all_records)
                raise

        if source_path.exists():
            source_path.unlink()
        return True

    def reindex_document(self, document_id: str) -> DocumentRecord | None:
        """Rebuild the FAISS index from stored source files and refresh chunk counts."""

        with self.session_factory() as session:
            records = list(session.scalars(select(DocumentRecord)).all())
            target = next((record for record in records if record.document_id == document_id), None)
            if target is None:
                return None

            chunk_counts = self._rebuild_index_for_records(records)
            for record in records:
                record.chunk_count = chunk_counts.get(record.document_id, record.chunk_count)
            session.commit()
            session.refresh(target)
            return target

    def document_count(self) -> int:
        """Return the number of indexed documents stored in SQLite."""

        with self.session_factory() as session:
            count = session.scalar(select(func.count(DocumentRecord.id)))
            return int(count or 0)
