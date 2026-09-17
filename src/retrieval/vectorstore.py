"""
Vector store build/load. Uses Chroma locally for Stage 1 — no external
infra required. Each chunk is stored with metadata (doc_id, file_name,
page_number, source_type, visibility) so results stay traceable back to a
source and filterable by the authorization layer in src/context.
"""
from __future__ import annotations

import logging

from langchain_chroma import Chroma

from src.config import get_path
from src.ingestion.preprocess import Chunk
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


_CACHED_VECTORSTORE: Chroma | None = None


def reset_vectorstore_cache() -> None:
    """Invalidates the in-memory Chroma vectorstore instance."""
    global _CACHED_VECTORSTORE
    _CACHED_VECTORSTORE = None


def build_vectorstore(chunks: list[Chunk], visibility: str = "shared") -> Chroma:
    """Embed chunks and ADD them to the existing local Chroma store.
    `visibility` tags every chunk (shared/client/firm/private) so retrieval
    can later filter by what the current user/session is authorized to see.
    Uses add_texts() on the existing store so previously ingested documents
    are NOT overwritten on each upload. Self-heals if database schema is corrupted.
    """
    global _CACHED_VECTORSTORE
    persist_dir = str(get_path("vectorstore_dir"))
    embeddings = get_embeddings()

    texts = [c.text for c in chunks]
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "file_name": c.file_name,
            "page_number": c.page_number,
            "source_type": c.source_type,
            "visibility": visibility,
        }
        for c in chunks
    ]
    ids = [c.chunk_id for c in chunks]

    logger.info("Adding %d chunks to vectorstore at %s", len(chunks), persist_dir)

    # Load (or create) the persistent store and ADD to it — never crash
    store = None
    if _CACHED_VECTORSTORE is not None:
        try:
            _CACHED_VECTORSTORE.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            store = _CACHED_VECTORSTORE
        except Exception as e:
            logger.warning("Failed to add to cached vectorstore (%s). Resetting cache.", e)
            _CACHED_VECTORSTORE = None

    if store is None:
        try:
            store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name="briefly_stage1",
            )
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.warning("Error with existing store (%s). Recreating clean vectorstore...", e)
            import shutil
            from pathlib import Path
            p = Path(persist_dir)
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
            store = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=persist_dir,
                collection_name="briefly_stage1",
            )

    _CACHED_VECTORSTORE = store
    try:
        count = store._collection.count()
        logger.info("Vectorstore now contains %d documents", count)
    except Exception:
        pass
    return store


def load_vectorstore() -> Chroma:
    global _CACHED_VECTORSTORE
    if _CACHED_VECTORSTORE is not None:
        return _CACHED_VECTORSTORE

    persist_dir = str(get_path("vectorstore_dir"))
    embeddings = get_embeddings()
    try:
        _CACHED_VECTORSTORE = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="briefly_stage1",
        )
    except Exception as e:
        logger.warning("Error loading vectorstore (%s). Initializing fresh vectorstore...", e)
        import shutil
        from pathlib import Path
        p = Path(persist_dir)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        _CACHED_VECTORSTORE = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="briefly_stage1",
        )
    return _CACHED_VECTORSTORE
