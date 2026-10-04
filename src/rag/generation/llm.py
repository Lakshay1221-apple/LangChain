"""Language model integration."""

"""LLM configuration for the RAG pipeline."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


MODEL_PROVIDER = "groq"
MODEL_NAME = "llama-3.1-8b-instant"


def create_llm() -> BaseChatModel:
    """Create and return the Groq chat model."""

    llm = init_chat_model(
        model_provider=MODEL_PROVIDER,
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
    )

    return llm