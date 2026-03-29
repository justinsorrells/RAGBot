"""LLM wrapper tests."""

from types import SimpleNamespace

from ragbot.config import Settings
from ragbot.core.llm import LangChainLLMClient, build_context_block, resolve_provider
from ragbot.core.retriever import RetrievedChunk


class StubChatModel:
    """Test double for chat model invocations."""

    def __init__(self) -> None:
        self.last_messages = None

    def invoke(self, input):
        self.last_messages = input
        return SimpleNamespace(content="Grounded answer")


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


def test_resolve_provider_prefers_openai_when_a_key_is_available() -> None:
    """Provider auto-selection should use OpenAI when a key is configured."""

    settings = Settings(openai_api_key="test-key", llm_provider="auto")
    assert resolve_provider(settings) == "openai"
