"""Hybrid retrieval implementation."""

from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import  EnsembleRetriever

def create_hybrid_retriever(
    bm25_retriever: BaseRetriever,
    vector_retriever: BaseRetriever,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> EnsembleRetriever:
    """Create a hybrid retriever using weighted rank fusion."""

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever,
        ],
        weights=[
            bm25_weight,
            vector_weight,
        ],
    )

    return hybrid_retriever