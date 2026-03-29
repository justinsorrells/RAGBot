"""Retriever tests."""

from ragbot.core.indexer import DocumentIndexer, PersistentVectorIndex
from ragbot.core.retriever import DocumentRetriever
from tests.helpers import FakeEmbeddings


def test_retriever_returns_the_most_relevant_chunk(tmp_path) -> None:
    """The retriever should rank semantically related chunks highest."""

    vector_index = PersistentVectorIndex(FakeEmbeddings(), tmp_path / "faiss_index")
    indexer = DocumentIndexer(vector_index)
    retriever = DocumentRetriever(vector_index)

    indexer.index_document(
        text="Apples and oranges are fruit often discussed in grocery stores.",
        filename="fruit.txt",
        document_id="doc-fruit",
    )
    indexer.index_document(
        text="Roadie supports same-day gig delivery and last-mile logistics workflows.",
        filename="roadie.txt",
        document_id="doc-roadie",
    )

    results = retriever.search("How does gig delivery work?", limit=2)

    assert results
    assert results[0].filename == "roadie.txt"
    assert "delivery" in results[0].text.lower()
    assert results[0].chunk_index == 0
