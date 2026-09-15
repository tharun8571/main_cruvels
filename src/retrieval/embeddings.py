"""
Embeddings factory. Kept separate from the vector store so the embedding
model can be swapped (e.g. local model instead of OpenAI) without touching
retrieval logic.
"""
from __future__ import annotations

from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings():
    settings = get_settings()
    provider = settings["embeddings"]["provider"]
    model = settings["embeddings"]["model"]

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
    elif provider == "openai":
        return OpenAIEmbeddings(model=model)

    raise ValueError(f"Unsupported embeddings provider: {provider}")
