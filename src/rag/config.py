"""Configuration for the RAG application."""

"""Application configuration."""

# Embedding
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector store
COLLECTION_NAME = "rag_documents"
CHROMA_DIR = "src/rag/chroma_db"

# Retrieval
VECTOR_TOP_K = 15
BM25_TOP_K = 15

# Reranking
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
RERANK_TOP_N = 5

# LLM
MODEL_PROVIDER = "groq"
MODEL_NAME = "openai/gpt-oss-20b"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 1024