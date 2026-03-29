"""Indexer tests."""

import pytest

from ragbot.core.indexer import CHUNK_SIZE, DocumentIndexer, PersistentVectorIndex, extract_text_from_bytes
from tests.helpers import FakeEmbeddings


def test_extract_text_from_txt_and_persist_index(tmp_path) -> None:
    """Large text uploads should be chunked and persisted into FAISS."""

    vector_index = PersistentVectorIndex(FakeEmbeddings(), tmp_path / "faiss_index")
    indexer = DocumentIndexer(vector_index)
    text = ("Roadie handles same-day delivery for local logistics teams. " * 60).strip()

    documents = indexer.index_document(text=text, filename="roadie.txt", document_id="doc-123")

    assert len(documents) > 1
    assert all(document.metadata["filename"] == "roadie.txt" for document in documents)
    assert all(document.metadata["document_id"] == "doc-123" for document in documents)
    assert all(len(document.page_content) <= CHUNK_SIZE + 50 for document in documents)

    reloaded_index = PersistentVectorIndex(FakeEmbeddings(), tmp_path / "faiss_index")
    matches = reloaded_index.similarity_search("same-day delivery", k=1)
    assert matches
    assert matches[0].metadata["filename"] == "roadie.txt"


def test_extract_text_from_bytes_validates_supported_extensions() -> None:
    """Unsupported file types should raise a clear validation error."""

    assert extract_text_from_bytes("notes.txt", b"  plain text contents  ") == "plain text contents"

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text_from_bytes("notes.md", b"# heading")
