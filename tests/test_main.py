"""Application assembly tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from ragbot.api import main as main_module
from ragbot.config import Settings
from tests.helpers import FakeEmbeddings, FakeLLMClient


class StubEngine:
    """Engine test double that tracks disposal."""

    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class HealthServiceStub:
    """Minimal service stub for lifespan tests."""

    def document_count(self) -> int:
        return 7


def test_build_rag_service_creates_a_working_runtime_graph(tmp_path, monkeypatch) -> None:
    """The runtime builder should wire the database, index, and service objects together."""

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'ragbot.db'}",
        faiss_index_path=str(tmp_path / "faiss_index"),
        document_store_path=str(tmp_path / "document_store"),
        llm_provider="ollama",
        default_top_k=2,
    )

    monkeypatch.setattr(main_module, "create_embeddings", lambda _: FakeEmbeddings())
    monkeypatch.setattr(main_module, "create_llm_client", lambda _: FakeLLMClient())

    service, engine = main_module.build_rag_service(settings)
    try:
        record = service.add_document("roadie.txt", b"Roadie helps merchants deliver packages locally.")
        listed = service.list_documents()

        assert record.filename == "roadie.txt"
        assert service.default_top_k == 2
        assert service.document_store_path == Path(settings.document_store_path)
        assert listed[0].document_id == record.document_id
    finally:
        engine.dispose()


def test_create_app_builds_service_during_lifespan_and_disposes_engine(monkeypatch) -> None:
    """App startup should build the runtime service and dispose the engine on shutdown."""

    settings = Settings(llm_provider="ollama")
    fake_engine = StubEngine()
    fake_service = HealthServiceStub()

    monkeypatch.setattr(main_module, "build_rag_service", lambda _: (fake_service, fake_engine))

    app = main_module.create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "documents_indexed": 7}
        assert client.app.state.rag_service is fake_service

    assert fake_engine.disposed is True
