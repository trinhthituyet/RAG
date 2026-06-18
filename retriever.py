"""Retrieval pipeline: vector lookup + LLM reranking."""

from config import MODEL, RETRIEVAL_K
from models import RankOrder, Result
from parsing import extract_json
from vector_store import VectorStore


class Reranker:
    """LLM-based reranker that reorders retrieved chunks by relevance."""

    SYSTEM_PROMPT = (
        "You are a document re-ranker.\n"
        "You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.\n"
        "The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.\n"
        "You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.\n"
        "Include all the chunk ids you are provided with, reranked.\n"
    )

    def __init__(self, llm, model: str = MODEL):
        self.llm = llm
        self.model = model

    def _make_user_prompt(self, question: str, chunks: list[Result]) -> str:
        prompt = (
            f"The user has asked the following question:\n\n{question}\n\n"
            "Order all the chunks of text by relevance to the question, from most relevant "
            "to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
            "Here are the chunks:\n\n"
        )
        for index, chunk in enumerate(chunks):
            prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
        prompt += (
            'Respond with ONLY a single JSON object and no other text, markdown, or code fences, '
            'of exactly this form:\n{"order": [3, 1, 2, ...]}\n'
            "including every chunk id you were given, reranked from most to least relevant."
        )
        return prompt

    def rerank(self, question: str, chunks: list[Result]) -> list[Result]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._make_user_prompt(question, chunks)},
        ]
        response = self.llm.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(extract_json(reply)).order
        # print(order)
        return [chunks[i - 1] for i in order]


class Retriever:
    """Combines vector-store lookup with LLM reranking."""

    def __init__(self, vector_store: VectorStore, reranker: Reranker, k: int = RETRIEVAL_K):
        self.vector_store = vector_store
        self.reranker = reranker
        self.k = k

    def fetch_unranked(self, question: str) -> list[Result]:
        return self.vector_store.query(question, k=self.k)

    def fetch(self, question: str) -> list[Result]:
        chunks = self.fetch_unranked(question)
        return self.reranker.rerank(question, chunks)
