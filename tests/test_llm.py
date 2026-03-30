"""LLM wrapper tests."""

from types import SimpleNamespace

import pytest

from ragbot.core import llm as llm_module
from ragbot.config import Settings
from ragbot.core.llm import LangChainLLMClient, build_context_block, create_embeddings, create_llm_client, resolve_provider
from ragbot.core.retriever import RetrievedChunk


class StubChatModel:
    """Test double for chat model invocations."""

    def __init__(self) -> None:
        self.last_messages = None

    def invoke(self, input):
        self.last_messages = input
        return SimpleNamespace(content="Grounded answer")


class ListContentChatModel:
    """Test double that returns list content."""

    def invoke(self, input):
        return SimpleNamespace(content=["Grounded", "answer"])


class RecordingInit:
    """Helper that records constructor kwargs."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_llm_client_builds_context_and_returns_string_response() -> None:
    """The wrapper should pass source context through to the underlying model."""

    model = StubChatModel()
    client = LangChainLLMClient(model)
    sources = [RetrievedChunk(filename="roadie.txt", text="Roadie enables same-day delivery.", chunk_index=0)]

    answer = client.answer_question("What does Roadie do?", sources)

    assert answer == "Grounded answer"
    assert "Roadie enables same-day delivery." in model.last_messages[1].content
    assert "Question: What does Roadie do?" in model.last_messages[1].content
    assert "[Source 1: roadie.txt]" in build_context_block(sources)


def test_llm_client_flattens_list_content() -> None:
    """List-style model responses should be converted into a simple string."""

    client = LangChainLLMClient(ListContentChatModel())
    answer = client.answer_question("What is Roadie?", [RetrievedChunk(filename="doc.txt", text="Roadie is logistics.", chunk_index=0)])

    assert answer == "Grounded answer"


def test_resolve_provider_prefers_openai_when_a_key_is_available() -> None:
    """Provider auto-selection should use OpenAI when a key is configured."""

    settings = Settings(openai_api_key="test-key", llm_provider="auto")
    assert resolve_provider(settings) == "openai"


def test_resolve_provider_uses_ollama_without_an_openai_key() -> None:
    """Auto provider selection should fall back to Ollama when no key is configured."""

    settings = Settings(llm_provider="auto")
    assert resolve_provider(settings) == "ollama"


def test_resolve_provider_validates_configuration_errors() -> None:
    """Invalid provider combinations should raise clear errors."""

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        resolve_provider(Settings(llm_provider="openai"))

    with pytest.raises(ValueError, match="must be one of"):
        resolve_provider(Settings(llm_provider="invalid-provider"))


def test_create_embeddings_uses_the_selected_provider(monkeypatch) -> None:
    """Embedding factory selection should map settings into provider constructors."""

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", RecordingInit)
    monkeypatch.setattr("langchain_ollama.OllamaEmbeddings", RecordingInit)

    openai_result = create_embeddings(
        Settings(
            llm_provider="openai",
            openai_api_key="test-key",
            openai_embedding_model="embed-model",
        )
    )
    ollama_result = create_embeddings(
        Settings(
            llm_provider="ollama",
            ollama_embedding_model="nomic-test",
            ollama_base_url="http://ollama.local",
        )
    )

    assert openai_result.kwargs == {"model": "embed-model", "api_key": "test-key"}
    assert ollama_result.kwargs == {"model": "nomic-test", "base_url": "http://ollama.local"}


def test_create_llm_client_uses_the_selected_provider(monkeypatch) -> None:
    """Chat model factory selection should wrap the selected provider in the client."""

    monkeypatch.setattr("langchain_openai.ChatOpenAI", RecordingInit)
    monkeypatch.setattr("langchain_ollama.ChatOllama", RecordingInit)

    openai_client = create_llm_client(
        Settings(
            llm_provider="openai",
            openai_api_key="test-key",
            openai_chat_model="chat-model",
        )
    )
    ollama_client = create_llm_client(
        Settings(
            llm_provider="ollama",
            ollama_chat_model="llama-test",
            ollama_base_url="http://ollama.local",
        )
    )

    assert isinstance(openai_client, llm_module.LangChainLLMClient)
    assert openai_client.chat_model.kwargs == {
        "model": "chat-model",
        "api_key": "test-key",
        "temperature": 0,
    }
    assert ollama_client.chat_model.kwargs == {
        "model": "llama-test",
        "base_url": "http://ollama.local",
        "temperature": 0,
    }
