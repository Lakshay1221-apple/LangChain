"""Embedding utilities."""

"""Embedding utilities for the RAG pipeline."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Create and return the embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    return embeddings


def embed_documents(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> list[list[float]]:
    """Generate embeddings for a list of documents."""

    texts = [document.page_content for document in documents]

    vectors = embeddings.embed_documents(texts)

    return vectors