"""End-to-end RAG pipeline."""

from langchain_core.documents import Document

try:
    from ..generation.llm import create_llm
    from ..generation.prompt import create_rag_prompt
    from ..retrieval.reranker import create_reranker, rerank_documents
except (ImportError, ValueError):
    from generation.llm import create_llm
    from generation.prompt import create_rag_prompt
    from retrieval.reranker import create_reranker, rerank_documents


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        retriever,
        reranker_top_n: int = 5,
        rerank_model: str = "ms-marco-MiniLM-L-12-v2",
    ):
        self.retriever = retriever
        self.reranker = create_reranker(top_n=reranker_top_n, model_name=rerank_model)
        self.prompt = create_rag_prompt()
        self.llm = create_llm()

    def retrieve(self, question: str) -> list[Document]:
        """Retrieve relevant documents with deduplication."""

        documents = self.retriever.invoke(question)

        # Deduplicate retrieved documents based on normalized content
        seen = set()
        unique_documents = []
        for doc in documents:
            content_key = " ".join(doc.page_content.split())
            if content_key not in seen:
                seen.add(content_key)
                unique_documents.append(doc)

        return unique_documents

    def rerank(
        self,
        documents: list[Document],
        question: str,
    ) -> list[Document]:
        """Rerank retrieved documents."""

        return rerank_documents(
            documents=documents,
            query=question,
            reranker=self.reranker,
        )

    def generate(
        self,
        question: str,
        documents: list[Document],
    ):
        """Generate an answer using the retrieved context."""

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        messages = self.prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = self.llm.invoke(messages)

        return response

    def invoke(self, question: str):
        """Run the complete RAG pipeline."""

        documents = self.retrieve(question)

        reranked_documents = self.rerank(
            documents,
            question,
        )

        response = self.generate(
            question,
            reranked_documents,
        )

        return response