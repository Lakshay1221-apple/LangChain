"""BM25 retrieval implementation."""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def create_bm25_retriever(documents: list[Document],
    k: int = 10,) -> BM25Retriever:
    ''' Create a BM25 retriever from a list of LangChain Documents.'''

    retriever = BM25Retriever.from_documents(
        documents=documents,
        k=k,
    )

    return retriever

    
