"""Service-layer tests."""

import pytest

from ragbot.core.service import RAGService
from ragbot.db.session import create_engine_from_url, create_session_factory, init_database
from tests.helpers import FakeLLMClient, build_test_service


class RebuildTrackingVectorIndex:
    """Vector index stub that tracks rebuild calls."""

    def __init__(self) -> None:
        self.rebuild_calls = []

    def rebuild(self, documents) -> None:
        self.rebuild_calls.append(list(documents))


class FailingIndexer:
    """Indexer stub that raises during indexing after a source file is stored."""

    def __init__(self) -> None:
        self.vector_index = RebuildTrackingVectorIndex()

    def index_document(self, text: str, filename: str, document_id: str):
        raise RuntimeError("Indexing failed")


class RetrieverStub:
    """Unused retriever stub for service construction."""

    def search(self, question: str, limit: int = 4):
        return []


def test_service_returns_missing_document_flags(test_settings) -> None:
    """Delete and reindex should return simple missing-document signals."""

    service, engine = build_test_service(test_settings)
    try:
        assert service.delete_document("missing-doc") is False
        assert service.reindex_document("missing-doc") is None
    finally:
        engine.dispose()


def test_service_reindex_raises_when_the_stored_source_file_is_missing(test_settings) -> None:
    """Reindexing should fail clearly when the stored source file was removed."""

    service, engine = build_test_service(test_settings)
    try:
        record = service.add_document("roadie.txt", b"Roadie supports local delivery.")
        service.stored_document_path(record.document_id, record.filename).unlink()

        with pytest.raises(ValueError, match="is missing"):
            service.reindex_document(record.document_id)
    finally:
        engine.dispose()


def test_service_cleans_up_stored_files_when_indexing_fails(tmp_path) -> None:
    """Failed ingestion should remove the stored source file and rebuild the index state."""

    engine = create_engine_from_url(f"sqlite:///{tmp_path / 'ragbot.db'}")
    init_database(engine)
    session_factory = create_session_factory(engine)
    indexer = FailingIndexer()
    service = RAGService(
        session_factory=session_factory,
        indexer=indexer,
        retriever=RetrieverStub(),
        llm_client=FakeLLMClient(),
        document_store_path=tmp_path / "document_store",
    )

    with pytest.raises(RuntimeError, match="Indexing failed"):
        service.add_document("broken.txt", b"Some text")

    assert list((tmp_path / "document_store").glob("*")) == []
    assert indexer.vector_index.rebuild_calls == [[]]
    engine.dispose()
