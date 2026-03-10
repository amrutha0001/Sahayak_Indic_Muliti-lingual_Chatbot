"""
app.py
------
FastAPI backend — RAG pipeline + live ingestion endpoints.

Endpoints:
  POST /chat            → main chat endpoint
  GET  /schemes         → list all loaded schemes
  GET  /health          → health check
  POST /ingest/file     → upload a PDF/DOCX/TXT and add to knowledge base
  POST /ingest/url      → crawl a URL and add to knowledge base

Run with:
    uvicorn app:app --reload --port 8000
"""

import os
import re

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

from rag import answer, collection

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Indic Government Scheme RAG Assistant",
    description="Multilingual answers about Indian government schemes using local LLM + RAG",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

SCHEMES_DIR = "./schemes"
SUPPORTED_UPLOAD_TYPES = {
    "text/plain", "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword", "text/markdown",
}
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc", ".md"}


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    language: str = "English"

class Source(BaseModel):
    scheme: str
    section: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    tier: str = "txt"      # "txt" | "url" | "web" | "none" 

class UrlIngestRequest(BaseModel):
    url: str


# ── Helper: re-embed a newly added file/URL without full rebuild ───────────────
def _add_to_collection(scheme_name: str, sections: dict, source_file: str):
    """Add new chunks to the existing ChromaDB collection."""
    docs, metas, ids = [], [], []
    for section_key, section_text in sections.items():
        if len(section_text.strip()) < 30:
            continue
        safe = re.sub(r"[^\w]", "_", source_file)[:60]
        chunk_id = f"{safe}::{section_key}"
        docs.append(section_text)
        metas.append({
            "scheme_name": scheme_name,
            "section":     section_key,
            "source_file": source_file,
        })
        ids.append(chunk_id)

    if docs:
        collection.add(documents=docs, metadatas=metas, ids=ids)
    return len(docs)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": collection.count()}


@app.get("/schemes")
def list_schemes():
    results = collection.get(include=["metadatas"])
    names = sorted(set(m["scheme_name"] for m in results["metadatas"]))
    return {"schemes": names, "total": len(names)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = answer(req.question, language=req.language)
    return ChatResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        tier=result.get("tier", "txt"),
    )


# /ingest/file endpoint removed — add files via schemes/ folder and re-run ingest.py


@app.post("/ingest/url")
def ingest_url_endpoint(req: UrlIngestRequest):
    """
    Crawl a URL, extract visible text, and embed into ChromaDB.
    """
    from ingest import extract_url, parse_text_to_sections, HAS_BS4

    if not HAS_BS4:
        raise HTTPException(
            status_code=503,
            detail="URL ingestion unavailable. Install: pip install beautifulsoup4"
        )

    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    title, text = extract_url(url)
    if not text:
        raise HTTPException(status_code=422, detail=f"Could not fetch or parse content from: {url}")

    sections = parse_text_to_sections(title, text)
    added    = _add_to_collection(title, sections, url)

    # Persist URL to urls.txt for future re-ingests
    os.makedirs(SCHEMES_DIR, exist_ok=True)
    urls_file = os.path.join(SCHEMES_DIR, "urls.txt")
    with open(urls_file, "a", encoding="utf-8") as f:
        f.write(url + "\n")

    return {
        "status": "ingested",
        "scheme_name": title,
        "url": url,
        "chunks_added": added,
        "total_chunks": collection.count(),
    }