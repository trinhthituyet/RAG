# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` (see `uv.lock`, `pyproject.toml`). Python is pinned to 3.12 (`.python-version`).

- `uv sync` — install dependencies (uses Apple's internal PyPI index for `apple-certifi`; will fail off-corp).
- `uv run main.py` — run the top-level RAG demo (currently launches the Gradio chat UI; uncomment `main()` in `main.py` to run a single hard-coded query instead).
- `uv run pro_implementation/ingest.py` — build `preprocessed_db/` with the "pro" pipeline (OpenAI embeddings + LLM-driven chunking, parallelized).
- `uv run implementation/ingest.py` — build `vector_db/` with the simpler LangChain baseline.
- `uv run evaluation/eval.py <test_row_number>` — evaluate a single test from `evaluation/tests.jsonl` (retrieval metrics + LLM-as-judge). The eval harness only exercises the `implementation/` baseline.
- To rebuild the top-level vector store, edit `main.py` to call `add_data_to_db()` instead of `main()`.

## Architecture

The repo contains **three parallel implementations** of the same Insurellm RAG demo, plus an eval harness. They share `knowledge-base/` (markdown files under `company/`, `contracts/`, `employees/`, `products/` — folder name becomes the `type` metadata) but diverge in everything else.

### Top-level package (recently refactored from a monolithic `main.py`)

Modules at repo root, all importing each other directly (no package — flat layout):

- `config.py` — constants, `load_dotenv`, `fetch_apple_token()` (shells out to `/usr/local/bin/appleconnect` for an OAuth token), `make_llm_client()` (points OpenAI client at `https://floodgate.g.apple.com/api/openai/v1`), `make_embeddings()` (HuggingFace `all-MiniLM-L6-v2`).
- `models.py` — Pydantic models (`Result`, `Chunk`, `Chunks`, `RankOrder`).
- `parsing.py` — `extract_json` for stripping prose / fences from LLM replies (the LLM gateway here doesn't support `response_format`).
- `documents.py` — `DocumentLoader`.
- `chunker.py` — `Chunker.chunk_all()` uses the LLM to split each doc into headline+summary+original chunks (~500 chars target, ~25% overlap).
- `vector_store.py` — `VectorStore` wraps a chromadb `PersistentClient` rooted at `preprocessed_db/`, collection `docs`. `rebuild()` deletes and re-creates; `query()` returns `Result` objects.
- `retriever.py` — `Retriever` does vector lookup (`RETRIEVAL_K=10`) then LLM reranks via `Reranker`.
- `pipeline.py` — `RAGPipeline.answer()` is rewrite-query → retrieve → rerank → answer. `chat()` is the Gradio-friendly variant returning only the answer text.
- `main.py` — `build_pipeline()` wires it all up; `add_data_to_db()` ingests; `main()` runs a sample query.

The top-level token fetch and embedding/chroma init are now lazy (created in factories), unlike the original `main.py` which initialized them at import time.

### `implementation/` — LangChain baseline

Uses `OpenAIEmbeddings("text-embedding-3-large")`, `RecursiveCharacterTextSplitter`, and `langchain_chroma`. Persists to **`vector_db/`** (different dir from the top-level pipeline). Model is `gpt-4.1-nano` via standard OpenAI. This is what `evaluation/eval.py` imports from.

### `pro_implementation/` — advanced pipeline

LLM-driven chunking (parallelized with `multiprocessing.Pool`, `tenacity` retries), OpenAI embeddings + LLM reranker, **dual-query retrieval** (original + rewritten, merged, then reranked, top `FINAL_K=10`). Persists to **`preprocessed_db/`**.

### `evaluation/`

`test.py` defines `TestQuestion` (question + keywords + reference_answer + category) and loads `tests.jsonl`. `eval.py` computes per-keyword **MRR** and **nDCG** plus an LLM-as-judge `AnswerEval` (accuracy/completeness/relevance, 1–5). Always evaluates the `implementation/` baseline — it does not touch the top-level or `pro_implementation` code paths.

## Important Gotchas

- **Two pipelines write to the same `preprocessed_db/` with incompatible embeddings.** The top-level pipeline writes HuggingFace `all-MiniLM-L6-v2` vectors (384-dim); `pro_implementation/` writes OpenAI `text-embedding-3-large` vectors (3072-dim). Running ingestion from one will break queries from the other. They are not interchangeable — pick one, rebuild the DB, and stick with it for that session.
- **The top-level pipeline depends on Apple corp infra.** `config.fetch_apple_token()` shells out to `appleconnect`, and the OpenAI client base URL is the internal `floodgate` gateway. Off-corp, only the `implementation/` and `pro_implementation/` paths (which use stock OpenAI) will work — and `implementation/` requires an `OPENAI_API_KEY` in `.env`.
- **The Apple gateway model used by the top-level pipeline doesn't support `response_format`** — that's why `parsing.extract_json` exists and chunks/rerank-orders are hand-parsed. `pro_implementation/` and `evaluation/` use `litellm.completion(..., response_format=Chunks)` directly, since they hit standard providers.
- **Mutable default `history=[]`** is still present in `implementation/answer.py` and `pro_implementation/answer.py`. The top-level pipeline was already cleaned up to use `None`.
- **`preprocessed_db/` is gitignored** but the eval harness expects it (or `vector_db/`) to be populated — run an ingest before evaluating.
