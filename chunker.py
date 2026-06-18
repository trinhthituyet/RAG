"""LLM-driven chunking of documents into Knowledge Base entries."""

from tqdm import tqdm

from config import AVERAGE_CHUNK_SIZE, MODEL
from models import Chunks, Result
from parsing import extract_json


class Chunker:
    """Splits documents into overlapping chunks via the LLM and validates with Pydantic."""

    def __init__(self, llm, model: str = MODEL, average_chunk_size: int = AVERAGE_CHUNK_SIZE):
        self.llm = llm
        self.model = model
        self.average_chunk_size = average_chunk_size

    def _make_prompt(self, document: dict) -> str:
        how_many = (len(document["text"]) // self.average_chunk_size) + 1
        return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:
<document>
{document["text"]}
</document>

Respond with ONLY a single JSON object and no other text, markdown, or code fences, of exactly this form:
{{"chunks": [{{"headline": "...", "summary": "...", "original_text": "..."}}]}}
where "original_text" is the exact text of the chunk taken verbatim from the document.
"""

    def chunk_document(self, document: dict) -> list[Result]:
        messages = [{"role": "user", "content": self._make_prompt(document)}]
        response = self.llm.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content
        reply_json = extract_json(reply)
        print(reply_json)
        doc_as_chunks = Chunks.model_validate_json(reply_json).chunks
        return [chunk.as_result(document) for chunk in doc_as_chunks]

    def chunk_all(self, documents: list[dict]) -> list[Result]:
        chunks: list[Result] = []
        for doc in tqdm(documents):
            chunks.extend(self.chunk_document(doc))
        return chunks
