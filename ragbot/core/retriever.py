"""Similarity search helpers."""

from dataclasses import dataclass

from ragbot.core.indexer import PersistentVectorIndex


@dataclass(slots=True)
class RetrievedChunk:
    """A retrieved source chunk returned to the API caller."""

    filename: str
    text: str
    chunk_index: int | None = None


class DocumentRetriever:
    """Run similarity search over indexed document chunks."""

    def __init__(self, vector_index: PersistentVectorIndex) -> None:
        self.vector_index = vector_index

    def search(self, question: str, limit: int = 4) -> list[RetrievedChunk]:
        """Return the top matching chunks for a user question."""

        matches = self.vector_index.similarity_search(question, k=limit)
        return [
            RetrievedChunk(
                filename=match.metadata.get("filename", "unknown"),
                text=match.page_content,
                chunk_index=match.metadata.get("chunk_index"),
            )
            for match in matches
        ]
