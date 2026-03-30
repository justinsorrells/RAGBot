"""Pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from ragbot.api.main import create_app
from ragbot.config import Settings
from tests.helpers import build_test_service


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Return isolated settings for each test."""

    return Settings(
        database_url=f"sqlite:///{tmp_path / 'ragbot.db'}",
        faiss_index_path=str(tmp_path / "faiss_index"),
        document_store_path=str(tmp_path / "document_store"),
        llm_provider="ollama",
        default_top_k=3,
    )


@pytest.fixture
def client(test_settings: Settings):
    """Return a test client backed by fake embeddings and fake LLMs."""

    service, engine = build_test_service(test_settings)
    app = create_app(settings=test_settings, service=service)
    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()
