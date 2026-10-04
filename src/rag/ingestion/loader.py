"""Document loading utilities."""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

def load_documents(data_path: str):
    """Load PDF and UTF-8 text documents from a directory."""

    data_directory = Path(data_path)
    documents = []

    pdf_loader = DirectoryLoader(
        str(data_directory),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents.extend(pdf_loader.load())

    text_loader = DirectoryLoader(
        str(data_directory),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(text_loader.load())

    return documents