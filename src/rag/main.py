"""RAG application entry point."""

import argparse
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


if __package__:
    from .config import BM25_TOP_K, RERANK_MODEL, RERANK_TOP_N, VECTOR_TOP_K
    from .ingestion.loader import load_documents
    from .ingestion.cleaner import clean_documents
    from .ingestion.chunker import chunk_documents
    from .embeddings.embedder import create_embedding_model
    from .embeddings.vector_store import create_vector_store
    from .pipeline.rag_pipeline import RAGPipeline
    from .retrieval.bm25_retriever import create_bm25_retriever
    from .retrieval.hybrid_retriever import create_hybrid_retriever
    from .retrieval.vector_retriever import create_vector_retriever
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import BM25_TOP_K, RERANK_MODEL, RERANK_TOP_N, VECTOR_TOP_K
    from ingestion.loader import load_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import chunk_documents
    from embeddings.embedder import create_embedding_model
    from embeddings.vector_store import create_vector_store
    from pipeline.rag_pipeline import RAGPipeline
    from retrieval.bm25_retriever import create_bm25_retriever
    from retrieval.hybrid_retriever import create_hybrid_retriever
    from retrieval.vector_retriever import create_vector_retriever


DEFAULT_QUERY = (
    "What were the major experiences and influences during Elon Musk's "
    "childhood that shaped his personality, interests, and later ambitions?"
)


def main(query: str = DEFAULT_QUERY) -> None:
    # ---------------------------------------------------------
    # 1. DATA PATH
    # ---------------------------------------------------------
    data_path = Path(__file__).resolve().parent / "data"

    # ---------------------------------------------------------
    # 2. DOCUMENT LOADING
    # ---------------------------------------------------------
    loaded_documents = load_documents(str(data_path))

    print(f"\nLoaded documents: {len(loaded_documents)}")

    # Show source distribution
    raw_sources = Counter(
        doc.metadata.get("title", "Unknown")
        for doc in loaded_documents
    )

    print("\nRaw:")
    for title, count in raw_sources.items():
        print(f"{title}: {count}")

    # ---------------------------------------------------------
    # 3. DOCUMENT CLEANING
    # ---------------------------------------------------------
    cleaned_documents = clean_documents(loaded_documents)

    print(f"\nCleaned documents: {len(cleaned_documents)}")
    print(
        f"Removed documents: "
        f"{len(loaded_documents) - len(cleaned_documents)}"
    )

    # Show cleaned source distribution
    cleaned_sources = Counter(
        doc.metadata.get("title", "Unknown")
        for doc in cleaned_documents
    )

    print("\nCleaned:")
    for title, count in cleaned_sources.items():
        print(f"{title}: {count}")

    # ---------------------------------------------------------
    # 4. CHUNKING
    # ---------------------------------------------------------
    chunks = chunk_documents(cleaned_documents)

    print(f"\nChunks created: {len(chunks)}")

    # ---------------------------------------------------------
    # 5. CHUNK SOURCE DISTRIBUTION
    # ---------------------------------------------------------
    chunk_sources = Counter(
        chunk.metadata.get("title", "Unknown")
        for chunk in chunks
    )

    print("\nChunks by source:")
    for title, count in chunk_sources.items():
        print(f"{title}: {count}")

    # ---------------------------------------------------------
    # 6. INSPECT SAMPLE CHUNKS
    # ---------------------------------------------------------
    print("\nSample chunks:")

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content[:500])
        print("\nMetadata:")
        print(chunk.metadata)

    # ---------------------------------------------------------
    # 7. EMBEDDINGS AND VECTOR STORE
    # ---------------------------------------------------------
    embedding_model = create_embedding_model()
    vector_store_path = Path(__file__).resolve().parent / "chroma_db"
    vector_store = create_vector_store(
        chunks,
        embedding_model,
        str(vector_store_path),
        rebuild=True,
    )

    stored_vectors = vector_store.get()["ids"]
    print(f"\nVectors stored: {len(stored_vectors)}")
    print(f"Vector store path: {vector_store_path}")

    # ---------------------------------------------------------
    # 8. RETRIEVAL, RERANKING, AND GENERATION
    # ---------------------------------------------------------
    vector_retriever = create_vector_retriever(vector_store, k=VECTOR_TOP_K)
    bm25_retriever = create_bm25_retriever(chunks, k=BM25_TOP_K)
    hybrid_retriever = create_hybrid_retriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
    )
    rag_pipeline = RAGPipeline(
        hybrid_retriever,
        reranker_top_n=RERANK_TOP_N,
        rerank_model=RERANK_MODEL,
    )

    print(f"\nQuestion: {query}")
    retrieved_documents = rag_pipeline.retrieve(query)
    reranked_documents = rag_pipeline.rerank(retrieved_documents, query)
    response = rag_pipeline.generate(query, reranked_documents)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:")
    for document in reranked_documents:
        title = document.metadata.get("title", "Unknown")
        source = document.metadata.get("source", "Unknown source")
        score = document.metadata.get("relevance_score", "N/A")
        print(f"- [{title}] (score: {score}) {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline.")
    parser.add_argument("query", nargs="*", help="Question to answer")
    arguments = parser.parse_args()
    main(" ".join(arguments.query) or DEFAULT_QUERY)