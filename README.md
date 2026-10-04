# LangChain & Advanced Hybrid RAG Ecosystem

A comprehensive repository showcasing modern LLM application development using **LangChain**, **LangChain Expression Language (LCEL)**, and a production-ready **Hybrid Retrieval-Augmented Generation (RAG)** pipeline.

---

## Architecture Overview

```
LangChain Project
├── src/rag/                        # Advanced Hybrid RAG Pipeline
│   ├── ingestion/                  # Document Loaders, Cleaners & Chunkers
│   ├── embeddings/                 # HuggingFace Embeddings & ChromaDB Vector Store
│   ├── retrieval/                  # BM25, Dense Vector, Hybrid Fusion & FlashRank Reranker
│   ├── generation/                 # Groq LLM Integration & Prompt Engineering
│   ├── pipeline/                   # End-to-End RAG Pipeline Orchestration
│   ├── config.py                   # Centralized Configuration
│   └── main.py                     # CLI & Application Entry Point
└── src/langchain/                  # Core LangChain & LCEL Learning Suite
    ├── models_langchain.ipynb      # Models, Structured Output & Tool Calling
    ├── messages_langchain.ipynb    # Chat Message Sequences & History
    ├── chains_langchain.ipynb      # LCEL Fundamentals
    ├── sequential_chain.ipynb      # Multi-step Sequential Workflows
    ├── parallel_chain.ipynb        # Parallel Execution with RunnableParallel
    ├── document_loaders.ipynb      # Document Loading Strategies
    ├── output_parser_langchain.ipynb # Pydantic & Output Parsing
    ├── langchain_cal_tool.ipynb    # Agent Tools & Function Calling
    └── langchain_context.ipynb     # Context Injection & Processing
```

---

## Key Features

### 1. Production-Grade Hybrid RAG Pipeline (`src/rag/`)
- **Multi-Source Ingestion & Sanitization**: Automated loading of unstructured PDFs and TXT files, rule-based text cleaning, metadata enrichment, and chunking with `RecursiveCharacterTextSplitter`.
- **Dense & Sparse Hybrid Retrieval**:
  - **Dense Vector Search**: Powered by `BAAI/bge-small-en-v1.5` embeddings and persistent `ChromaDB`.
  - **Sparse Keyword Search**: BM25 retriever utilizing `rank-bm25`.
  - **Reciprocal Rank Fusion**: Weighted ensemble retrieval balancing semantic similarity and exact keyword relevance.
- **Cross-Encoder Reranking**: Ultra-fast reranking using FlashRank (`ms-marco-MiniLM-L-12-v2`) to eliminate false positives and noise from multi-document collections.
- **Deduplication & Grounded Generation**: Content-level deduplication prevents context clutter; grounded system prompts ensure accurate answers with zero hallucinations via Groq's high-throughput LLMs.

### 2. LangChain & LCEL Interactive Suite (`src/langchain/`)
- Hands-on Jupyter notebooks covering foundational to advanced LangChain concepts.
- LCEL composition patterns using the pipe (`|`) operator.
- Deterministic vs. creative model routing, structured Pydantic output parsing, and dynamic tool calling.

---

## Requirements

- **Python**: 3.12+
- **Package Manager**: `uv` or `pip`
- **API Keys**:
  - [Groq API Key](https://console.groq.com/) (for RAG pipeline generation)
  - [Google Gemini API Key](https://aistudio.google.com/) (for notebook workflows)

---

## Quick Start & Setup

### 1. Clone & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Lakshay1221-apple/LangChain.git
cd LangChain

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Running the Hybrid RAG System

Run the RAG pipeline directly with the default query or pass your own question via CLI:

```bash
# Run with default query
python -m src.rag.main

# Run with a custom query
python -m src.rag.main "What were the major experiences during Elon Musk's childhood?"
```

### RAG Configuration (`src/rag/config.py`)

Key parameters can be customized in `src/rag/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `VECTOR_TOP_K` | `15` | Candidates retrieved via dense vector search |
| `BM25_TOP_K` | `15` | Candidates retrieved via BM25 sparse search |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank cross-encoder reranker model |
| `RERANK_TOP_N` | `5` | Top reranked passages passed to LLM context |
| `MODEL_NAME` | `openai/gpt-oss-20b` | Groq chat model for grounded response generation |

---

## Interactive Notebook Guide

| Notebook | Topic & Focus |
|---|---|
| [`models_langchain.ipynb`](src/langchain/models_langchain.ipynb) | Model initialization, temperature control, chat messages, and structured outputs |
| [`messages_langchain.ipynb`](src/langchain/messages_langchain.ipynb) | Working with `SystemMessage`, `HumanMessage`, `AIMessage`, and message history |
| [`chains_langchain.ipynb`](src/langchain/chains_langchain.ipynb) | Fundamental LCEL prompt-to-model-to-parser pipelines |
| [`sequential_chain.ipynb`](src/langchain/sequential_chain.ipynb) | Chaining multiple prompts sequentially where output feeds subsequent inputs |
| [`parallel_chain.ipynb`](src/langchain/parallel_chain.ipynb) | Multi-branch parallel execution using `RunnableParallel` |
| [`document_loaders.ipynb`](src/langchain/document_loaders.ipynb) | Loading text, PDF, and web documents into LangChain Document formats |
| [`output_parser_langchain.ipynb`](src/langchain/output_parser_langchain.ipynb) | Pydantic and JSON output parsers for guaranteed structured schemas |
| [`langchain_cal_tool.ipynb`](src/langchain/langchain_cal_tool.ipynb) | Function calling and tool definitions for AI agents |
| [`langchain_context.ipynb`](src/langchain/langchain_context.ipynb) | Dynamic context injection and prompt composition |

---

## License

This project is licensed under the MIT License.
