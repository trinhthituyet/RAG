"""Entry point: wires the RAG components together for ingestion and querying."""

from config import make_embeddings, make_llm_client
from chunker import Chunker
from documents import DocumentLoader
from pipeline import RAGPipeline
from retriever import Reranker, Retriever
from vector_store import VectorStore


def build_pipeline() -> RAGPipeline:
    llm = make_llm_client()
    embeddings = make_embeddings()
    vector_store = VectorStore(embeddings)
    retriever = Retriever(vector_store, Reranker(llm))
    return RAGPipeline(llm, retriever)


def add_data_to_db() -> None:
    llm = make_llm_client()
    embeddings = make_embeddings()
    documents = DocumentLoader().load()
    chunks = Chunker(llm).chunk_all(documents)
    VectorStore(embeddings).rebuild(chunks)


def main() -> None:
    pipeline = build_pipeline()
    answer, chunks = pipeline.answer("Who won the IIOTY award?")
    print(answer)
    print(chunks)


if __name__ == "__main__":
    # main()
    # add_data_to_db()
    import gradio as gr
    gr.ChatInterface(build_pipeline().chat).launch()
