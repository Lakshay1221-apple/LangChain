"""RAG application entry point."""

from collections import Counter
from pathlib import Path

if __package__:
    from .ingestion.loader import load_documents
    from .ingestion.cleaner import clean_documents
    from .ingestion.chunker import chunk_documents
else:
    from ingestion.loader import load_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import chunk_documents


def main() -> None:
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


if __name__ == "__main__":
    main()