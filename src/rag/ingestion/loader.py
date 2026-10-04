"""Document loading utilities."""

from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

def load_documents(data_path : str):
    ''' Load documents from a directory. '''
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    return documents