"""Result reranking utilities."""

from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document



def create_reranker(top_n : int = 5) -> FlashrankRerank:
    """Create and return a reranker instance."""

    reranker = FlashrankRerank(
        top_n = top_n
    )

    return reranker


def rerank_documents(
        query : str,
        documents : list[Document],
        reranker : FlashrankRerank,
) -> list[Document]:

    """ Rerank a list of documents based on a query using the provided reranker """

    reranked_documents = reranker.compress_documents(
        documents,
        query,
    )

    return reranked_documents

    
    