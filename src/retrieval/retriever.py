"""
Retrieval: fetch the most relevant chunks for a user question, filtered to
only what the current session is authorized to see (see src/context).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import get_settings
from src.context.permissions import AuthorizedScope
from .vectorstore import load_vectorstore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    file_name: str
    page_number: int
    score: float
    source_type: str


def retrieve_chunks(query: str, scope: AuthorizedScope) -> list[RetrievedChunk]:
    settings = get_settings()
    top_k = settings["retrieval"]["top_k"]
    threshold = settings["retrieval"]["score_threshold"]

    store = load_vectorstore()

    try:
        # Chroma similarity_search_with_relevance_scores returns (Document, score)
        results = store.similarity_search_with_relevance_scores(
            query,
            k=min(top_k * 2, 10),
        )
    except Exception as e:
        logger.warning("Error during similarity search: %s", e)
        return []

    filtered: list[RetrievedChunk] = []
    for doc, score in results:
        meta = doc.metadata
        if not scope.can_access(meta.get("visibility", "shared"), meta.get("doc_id")):
            continue
        if score < threshold:
            continue
        filtered.append(
            RetrievedChunk(
                text=doc.page_content,
                doc_id=meta["doc_id"],
                file_name=meta["file_name"],
                page_number=meta["page_number"],
                score=score,
                source_type=meta["source_type"],
            )
        )
        if len(filtered) >= top_k:
            break

    logger.info("Retrieved %d chunks above threshold %.2f for query: %s", len(filtered), threshold, query)
    return filtered
