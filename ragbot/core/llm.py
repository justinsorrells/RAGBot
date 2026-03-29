"""LLM and embeddings provider configuration."""

from typing import Any, Protocol, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from ragbot.config import Settings
from ragbot.core.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer questions using only the supplied "
    "context. If the answer is not present in the context, say that you do not know."
)


class ChatInvoker(Protocol):
    """Protocol for chat model invocations."""

    def invoke(self, input: Any) -> Any:
        """Invoke the underlying chat model."""


def resolve_provider(settings: Settings) -> str:
    """Resolve the active provider from configuration."""

    provider = settings.llm_provider.strip().lower()
    if provider == "auto":
        return "openai" if settings.openai_api_key else "ollama"
    if provider == "openai" and not settings.openai_api_key:
        raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")
    if provider not in {"openai", "ollama"}:
        raise ValueError("LLM_PROVIDER must be one of: auto, openai, ollama.")
    return provider


def build_context_block(sources: Sequence[RetrievedChunk]) -> str:
    """Format retrieved source chunks into a prompt-friendly block."""

    return "\n\n".join(
        f"[Source {index + 1}: {source.filename}]\n{source.text}"
        for index, source in enumerate(sources)
    )


class LangChainLLMClient:
    """LLM client that wraps a LangChain chat model."""

    def __init__(self, chat_model: ChatInvoker) -> None:
        self.chat_model = chat_model

    def answer_question(self, question: str, sources: Sequence[RetrievedChunk]) -> str:
        """Generate an answer grounded in the retrieved source chunks."""

        context = build_context_block(sources)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
        ]
        response = self.chat_model.invoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return " ".join(str(part) for part in content).strip()
        return str(content).strip()


def create_embeddings(settings: Settings) -> Embeddings:
    """Create the configured embeddings provider."""

    provider = resolve_provider(settings)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def create_llm_client(settings: Settings) -> LangChainLLMClient:
    """Create the configured LLM client."""

    provider = resolve_provider(settings)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        chat_model = ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama

        chat_model = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    return LangChainLLMClient(chat_model)
