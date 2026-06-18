"""End-to-end RAG pipeline: query rewriting, retrieval, and answer generation."""

from config import MODEL
from models import Result
from retriever import Retriever


SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


class RAGPipeline:
    """Orchestrates query rewrite → retrieval → answer generation."""

    def __init__(self, llm, retriever: Retriever, model: str = MODEL):
        self.llm = llm
        self.retriever = retriever
        self.model = model

    def rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        """Refine the user's question into a Knowledge Base search query."""
        history = history or []
        message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
        messages = [{"role": "user", "content": message}]
        response = self.llm.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content

    def _make_messages(self, question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
        context = "\n\n".join(
            f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
        )
        system_prompt = SYSTEM_PROMPT.format(context=context)
        return [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": question}
        ]

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
        """Answer a question using RAG; returns (answer, retrieved chunks)."""
        history = history or []
        query = self.rewrite_query(question, history)
        print(query)
        chunks = self.retriever.fetch(query)
        messages = self._make_messages(question, history, chunks)
        response = self.llm.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content, chunks

    def chat(self, question: str, history: list[dict] | None = None) -> str:
        """Gradio-friendly entry point that returns just the answer text."""
        answer, _ = self.answer(question, history)
        return answer
