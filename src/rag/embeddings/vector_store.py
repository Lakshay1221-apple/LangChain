'''Vector store utilities for the RAG pipeline.'''

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_NAME = "rag_documents"


def create_vector_store(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str,
    rebuild: bool = True,
) -> Chroma:
    """Create and persist a Chroma vector store."""
    import chromadb

    client = chromadb.PersistentClient(path=persist_directory)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
        client=client,
    )

    return vector_store


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str,
) -> Chroma:
    """Load an existing Chroma vector store."""

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    return vector_store