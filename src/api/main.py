"""
FastAPI Backend Server for Cruvels AI Legal Knowledge Assistant
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import get_settings, get_path, setup_logging
from src.context.permissions import AuthorizedScope
from src.agent.graph import run_agent
from src.ingestion.loader import load_document
from src.ingestion.preprocess import chunk_document
from src.retrieval.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)

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

    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Automatically ingest uploaded file
    try:
        doc = load_document(target_path)
        chunks = chunk_document(doc)
        if chunks:
            build_vectorstore(chunks, visibility="shared")
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
