"""Pydantic data models used across the RAG pipeline."""

from pydantic import BaseModel, Field


class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: dict) -> Result:
        page_content = f"{self.headline}\n\n{self.summary}\n\n{self.original_text}"
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(page_content=page_content, metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )
