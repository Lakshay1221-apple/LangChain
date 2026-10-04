"""Prompt construction utilities."""

from langchain_core.prompts import ChatPromptTemplate


RAG_SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions using the
provided context.

Follow these rules:
1. Use the provided context to answer the question.
2. Do not make up information that is not supported by the context.
3. If the answer cannot be found in the context, clearly say
   that you do not have enough information.
4. Keep the answer clear, accurate, and concise.

Context:
{context}
"""


def create_rag_prompt() -> ChatPromptTemplate:
    """Create the prompt template for the RAG pipeline."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    return prompt