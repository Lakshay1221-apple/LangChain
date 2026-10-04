"""Document chunking utilities."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(
        documents: list[Document],
        chunk_size: int = 500,
        overlap_size: int = 50,
) -> list[Document]:

    ''' Chunk a list of LangChain Documents into smaller chunks.'''

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    return chunks

