"""
FastAPI Backend Server for Cruvels AI Legal Knowledge Assistant
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from src.config import get_settings, get_path, setup_logging
from src.context.permissions import AuthorizedScope
from src.agent.graph import run_agent
from src.ingestion.loader import load_document
from src.ingestion.preprocess import chunk_document
from src.llm.model import get_llm
from src.retrieval.vectorstore import build_vectorstore, reset_vectorstore_cache

logger = logging.getLogger(__name__)

SUGGESTIONS_CACHE = {
    "key": None,
    "suggestions": None,
}

DEFAULT_SUGGESTIONS = [
    {
        "label": "Parties in the agreement",
        "question": "Who are the parties involved in this agreement?",
        "icon": "fa-solid fa-users",
    },
    {
        "label": "Key terms & obligations",
        "question": "What are the primary terms, duties, and obligations specified in this document?",
        "icon": "fa-solid fa-file-contract",
    },
    {
        "label": "Confidentiality & restrictions",
        "question": "What confidentiality terms, restrictive covenants, or non-disclosure obligations are stated?",
        "icon": "fa-solid fa-shield-halved",
    },
    {
        "label": "Governing law & jurisdiction",
        "question": "What is the governing law, dispute resolution process, and jurisdiction?",
        "icon": "fa-solid fa-gavel",
    },
]

app = FastAPI(
    title="Cruvels AI Legal Assistant API",
    description="Agentic Legal Knowledge RAG System with Cruvels & Kimi LLM",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    visibility: Optional[str] = "shared"


class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    is_fallback: bool


@app.get("/api/health")
def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "llm_provider": settings["llm"]["provider"],
        "llm_model": settings["llm"]["model"],
        "embedding_model": settings["embeddings"]["model"],
    }


@app.post("/api/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info("Received query: %s", req.question)
    scope = AuthorizedScope.firm_member(user_id="ui-user", case_id="ui-case")

    try:
        outcome = run_agent(req.question, scope)
        return AskResponse(
            answer=outcome.get("answer", ""),
            sources=outcome.get("sources", []),
            is_fallback=outcome.get("is_fallback", False),
        )
    except Exception as e:
        logger.exception("Error during RAG execution")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/suggestions")
def get_document_suggestions():
    """Dynamically generate tailored suggested questions by analyzing ingested documents."""
    global SUGGESTIONS_CACHE
    raw_dir = get_path("raw_data_dir")
    if not raw_dir.exists():
        return {"suggestions": DEFAULT_SUGGESTIONS, "is_dynamic": False}

    supported = get_settings()["ingestion"]["supported_extensions"]
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]
    if not files:
        return {"suggestions": DEFAULT_SUGGESTIONS, "is_dynamic": False}

    # Generate a cache key from filenames and file modification times
    cache_key = "-".join(f"{f.name}:{f.stat().st_mtime}" for f in sorted(files, key=lambda x: x.name))
    if SUGGESTIONS_CACHE["key"] == cache_key and SUGGESTIONS_CACHE["suggestions"]:
        return {"suggestions": SUGGESTIONS_CACHE["suggestions"], "is_dynamic": True}

    # Extract sample text from the available documents
    doc_text_snippets = []
    for f in files[:2]:
        try:
            doc = load_document(f)
            snippet = doc.full_text[:3000].strip()
            if snippet:
                doc_text_snippets.append(f"--- Document: {f.name} ---\n{snippet}")
        except Exception as e:
            logger.warning("Could not extract sample text from %s: %s", f.name, e)

    if not doc_text_snippets:
        return {"suggestions": DEFAULT_SUGGESTIONS, "is_dynamic": False}

    combined_text = "\n\n".join(doc_text_snippets)

    system_prompt = (
        "You are an expert legal document analyst. "
        "Based on the provided excerpt of the uploaded legal document(s), generate exactly 4 distinct, highly relevant, and specific questions a legal counsel or client would ask to understand this document. "
        "Return ONLY a valid JSON array of 4 objects with this exact structure:\n"
        "[\n"
        '  {"label": "Short label (3-6 words)", "question": "Full actionable legal question", "icon": "fa-solid fa-users"},\n'
        '  {"label": "Short label (3-6 words)", "question": "Full actionable legal question", "icon": "fa-solid fa-calendar-check"},\n'
        '  {"label": "Short label (3-6 words)", "question": "Full actionable legal question", "icon": "fa-solid fa-shield-halved"},\n'
        '  {"label": "Short label (3-6 words)", "question": "Full actionable legal question", "icon": "fa-solid fa-gavel"}\n'
        "]\n"
        "Valid icons include: fa-solid fa-users, fa-solid fa-calendar, fa-solid fa-clock, fa-solid fa-shield-halved, fa-solid fa-gavel, fa-solid fa-file-contract, fa-solid fa-handshake, fa-solid fa-money-bill-wave, fa-solid fa-building-shield, fa-solid fa-scale-balanced."
    )

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Document excerpt:\n{combined_text[:3500]}\n\nGenerate 4 tailored question suggestions in JSON format."),
        ]
        response = llm.invoke(messages)
        content = response.content.strip()

        # Clean JSON markdown if wrapped in ```json ... ```
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) >= 2:
            suggestions = []
            for item in parsed[:4]:
                if isinstance(item, dict) and "question" in item:
                    label = item.get("label") or item["question"][:35] + "..."
                    icon = item.get("icon") or "fa-solid fa-file-lines"
                    suggestions.append({
                        "label": label,
                        "question": item["question"],
                        "icon": icon,
                    })
            if suggestions:
                SUGGESTIONS_CACHE["key"] = cache_key
                SUGGESTIONS_CACHE["suggestions"] = suggestions
                return {"suggestions": suggestions, "is_dynamic": True}
    except Exception as e:
        logger.warning("Failed to dynamically generate document suggestions: %s", e)

    return {"suggestions": DEFAULT_SUGGESTIONS, "is_dynamic": False}


@app.get("/api/documents")
def list_documents():
    raw_dir = get_path("raw_data_dir")
    if not raw_dir.exists():
        return {"documents": []}

    files = []
    supported = get_settings()["ingestion"]["supported_extensions"]
    for p in raw_dir.iterdir():
        if p.is_file() and p.suffix.lower() in supported:
            files.append({
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "extension": p.suffix.lower(),
            })
    return {"documents": files}


@app.delete("/api/knowledge-base")
def clear_knowledge_base():
    """Wipe all ingested documents and the vector store."""
    global SUGGESTIONS_CACHE
    SUGGESTIONS_CACHE["key"] = None
    SUGGESTIONS_CACHE["suggestions"] = None
    reset_vectorstore_cache()
    errors = []

    # 1. Delete all files in the raw data directory
    raw_dir = get_path("raw_data_dir")
    if raw_dir.exists():
        for p in raw_dir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception as e:
                errors.append(f"raw/{p.name}: {e}")

    # 2. Wipe the vector store directory entirely
    vectorstore_dir = get_path("vectorstore_dir")
    if vectorstore_dir.exists():
        try:
            shutil.rmtree(vectorstore_dir)
            vectorstore_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"vectorstore: {e}")

    if errors:
        logger.warning("Clear knowledge base completed with errors: %s", errors)
        return {"status": "partial", "errors": errors}

    logger.info("Knowledge base cleared successfully")
    return {"status": "cleared"}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global SUGGESTIONS_CACHE
    SUGGESTIONS_CACHE["key"] = None
    SUGGESTIONS_CACHE["suggestions"] = None
    supported = get_settings()["ingestion"]["supported_extensions"]
    ext = Path(file.filename).suffix.lower()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported types: {supported}"
        )

    raw_dir = get_path("raw_data_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / file.filename

    contents = await file.read()
    with open(target_path, "wb") as f:
        f.write(contents)

    # Automatically ingest uploaded file
    try:
        doc = load_document(target_path)
        chunks = chunk_document(doc)
        if chunks:
            build_vectorstore(chunks, visibility="shared")
        logger.info("Successfully ingested %s with %d chunks", file.filename, len(chunks))
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": len(chunks),
        }
    except Exception as e:
        logger.exception("Error ingesting file %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# Serve static web frontend
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
