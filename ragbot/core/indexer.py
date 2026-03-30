"""Document parsing, chunking, and vector indexing."""

from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def extract_text_from_bytes(filename: str, content: bytes) -> str:
    """Extract normalized text from a supported uploaded document."""

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Only .txt and .pdf files are allowed.")

    if extension == ".txt":
        text = content.decode("utf-8", errors="ignore")
    else:
        reader = PdfReader(BytesIO(content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

    normalized = text.strip()
    if not normalized:
        raise ValueError("The uploaded document did not contain readable text.")

    return normalized


class PersistentVectorIndex:
    """Thin FAISS wrapper that persists the index to disk."""

    def __init__(self, embeddings: Embeddings, index_path: str | Path) -> None:
        self.embeddings = embeddings
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._store: FAISS | None = None
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        index_file = self.index_path / "index.faiss"
        store_file = self.index_path / "index.pkl"
        if index_file.exists() and store_file.exists():
            self._store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def _clear_persisted_index(self) -> None:
        for filename in ("index.faiss", "index.pkl"):
            file_path = self.index_path / filename
            if file_path.exists():
                file_path.unlink()

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the FAISS index and persist the updated store."""

        if not documents:
            return

        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(list(documents), self.embeddings)
            else:
                self._store.add_documents(list(documents))
            self._store.save_local(str(self.index_path))

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Run a similarity search against the local vector index."""

        with self._lock:
            if self._store is None:
                return []
            return self._store.similarity_search(query, k=k)

    def rebuild(self, documents: Sequence[Document]) -> None:
        """Replace the current FAISS index contents and persist the new state."""

        with self._lock:
            if not documents:
                self._store = None
                self._clear_persisted_index()
                return

            self._store = FAISS.from_documents(list(documents), self.embeddings)
            self._store.save_local(str(self.index_path))


class DocumentIndexer:
    """Chunk documents and write them into the vector index."""

    def __init__(
        self,
        vector_index: PersistentVectorIndex,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.vector_index = vector_index
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text: str, filename: str, document_id: str) -> list[Document]:
        """Split raw text into LangChain documents with source metadata."""

        chunks = [chunk.strip() for chunk in self.splitter.split_text(text) if chunk.strip()]
        return [
            Document(
                page_content=chunk,
                metadata={
                    "filename": filename,
                    "source": filename,
                    "document_id": document_id,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

    def index_document(self, text: str, filename: str, document_id: str) -> list[Document]:
        """Split and index a document, returning the generated chunks."""

        documents = self.split_text(text=text, filename=filename, document_id=document_id)
        if not documents:
            raise ValueError("The uploaded document did not produce any indexable chunks.")
        self.vector_index.add_documents(documents)
        return documents
