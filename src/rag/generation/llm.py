"""LLM configuration for the RAG pipeline."""

from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

try:
    from ..config import MAX_OUTPUT_TOKENS, MODEL_NAME, TEMPERATURE
except (ImportError, ValueError):
    from config import MAX_OUTPUT_TOKENS, MODEL_NAME, TEMPERATURE


def create_llm() -> BaseChatModel:
    """Create and return the Groq chat model."""

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return llm