"""Vector retrieval implementation."""

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


def create_vector_retriever(
    vector_store: Chroma,
    k: int = 10,
) -> BaseRetriever:
    """Create a vector retriever from a Chroma vector store."""

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )

    return retriever