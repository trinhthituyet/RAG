"""Configuration, environment, and factory helpers for clients/embeddings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(override=True)

MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
DB_NAME = "preprocessed_db"
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = Path("knowledge-base")
AVERAGE_CHUNK_SIZE = 500
RETRIEVAL_K = 10

LLM_BASE_URL = "https://floodgate.g.apple.com/api/openai/v1"
_TOKEN_CMD = (
    "/usr/local/bin/appleconnect getToken "
    "-C hvys3fcwcteqrvw3qzkvtk86viuoqv "
    "--token-type=oauth --interactivity-type=none "
    "-E prod -G pkce "
    "-o openid,dsid,accountname,profile,groups"
)


def fetch_apple_token() -> str:
    return os.popen(_TOKEN_CMD).read().split()[-1]


def make_llm_client() -> OpenAI:
    client = OpenAI(api_key=fetch_apple_token())
    client.base_url = LLM_BASE_URL
    return client


def make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
