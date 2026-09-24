"""FastAPI application: chat with the circulars, upload new ones, audit the index.

Endpoints
---------
GET  /                      chat UI (single page, no build step)
GET  /api/health            liveness + active configuration + index stats
POST /api/chat              ask a question -> cited answer
POST /api/search            raw retrieval (no generation) - useful for debugging
POST /api/documents/upload  add a new circular (file) and re-index immediately
POST /api/documents/text    add a new circular (pasted text)
GET  /api/documents         list documents, versions and supersession links
GET  /api/documents/{id}    one document with its full version history
GET  /api/chunks/{id}       fetch a single chunk (citation drill-down)
GET  /api/sessions          chat history
POST /api/reindex           rebuild the retrieval index from the DB
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    HealthOut,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SessionOut,
)
from core.config import settings
from core.db import get_session, init_db, session_scope
from core.logging_conf import get_logger
from core.models import (
    STATUS_ACTIVE,
    Chunk,
    ChatMessage,
    ChatSession,
    Document,
    DocumentVersion,
)
from generation.answer import generate_answer
from generation.llm import get_llm
from ingestion.embed import get_embedder
from ingestion.parse import build_parsed
from ingestion.pipeline import ingest_parsed, ingest_upload
from retrieval.rerank import get_reranker
from retrieval.service import get_retrieval_service

log = get_logger(__name__)
VERSION = "1.0.0"
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    svc = get_retrieval_service()
    svc.mark_dirty()
    try:
        svc.ensure_index()
        log.info("Startup complete. Index: %s", svc.stats())
    except Exception as exc:  # pragma: no cover - never block startup
        log.warning("Index warm-up failed: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version=VERSION,
    description="RAG over RBI Master Circulars with citations, amendment tracking and IR evaluation.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------- misc
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    svc = get_retrieval_service()
    try:
        stats = svc.stats()
    except Exception as exc:  # pragma: no cover
        stats = {"error": str(exc)}
    return HealthOut(
        status="ok",
        version=VERSION,
        environment=settings.environment,
        embedding=get_embedder().signature,
        reranker=get_reranker().name,
        llm=f"{get_llm().provider}:{settings.llm_model}",
        retrieval_mode=settings.retrieval_mode,
        chunker=settings.chunker,
        index=stats,
    )


# --------------------------------------------------------------------- chat
def _session_title(text: str) -> str:
    text = " ".join(text.split())
    return text[:60] + ("…" if len(text) > 60 else "")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    svc = get_retrieval_service()
    chunks, debug = svc.search(
        req.message,
        mode=req.mode,
        top_k=req.top_k,
        include_deprecated=req.include_deprecated,
        session=session,
    )
    result = generate_answer(
        req.message,
        chunks,
        provider=req.provider,
        model=req.model,
        use_llm_judge=req.use_llm_judge,
    )

    chat_session = None
    if req.session_id:
        chat_session = session.get(ChatSession, req.session_id)
    if chat_session is None:
        chat_session = ChatSession(title=_session_title(req.message))
        session.add(chat_session)
        session.flush()

    session.add(
        ChatMessage(session_id=chat_session.id, role="user", content=req.message)
    )
    assistant = ChatMessage(
        session_id=chat_session.id,
        role="assistant",
        content=result.answer,
        citations_json=json.dumps(result.citations),
        meta_json=json.dumps(
            {
                "retrieval": debug.as_dict(),
                "faithfulness": result.faithfulness.as_dict(),
                "usage": result.as_dict()["usage"],
            }
        ),
    )
    session.add(assistant)
    session.flush()

    payload = result.as_dict()
    return ChatResponse(
        session_id=chat_session.id,
        message_id=assistant.id,
        question=req.message,
        answer=payload["answer"],
        citations=payload["citations"],
        faithfulness=payload["faithfulness"],
        retrieval=debug.as_dict(),
        usage=payload["usage"],
        notices=payload["notices"],
    )


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest, session: Session = Depends(get_session)) -> SearchResponse:
    svc = get_retrieval_service()
    chunks, debug = svc.search(
        req.query,
        mode=req.mode,
        top_k=req.top_k,
        include_deprecated=req.include_deprecated,
        session=session,
    )
    return SearchResponse(
        query=req.query, results=[c.as_dict() for c in chunks], retrieval=debug.as_dict()
    )


# ---------------------------------------------------------------- documents
def _document_out(session: Session, doc: Document) -> DocumentOut:
    versions = list(
        session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == doc.id)
            .order_by(DocumentVersion.version_number)
        )
    )
    return DocumentOut(
        id=doc.id,
        doc_number=doc.doc_number,
        title=doc.title,
        regulator=doc.regulator,
        category=doc.category,
        source_url=doc.source_url,
        versions=[
            {
                "id": v.id,
                "version_number": v.version_number,
                "status": v.status,
                "published_date": v.published_date.isoformat() if v.published_date else None,
                "effective_date": v.effective_date.isoformat() if v.effective_date else None,
                "chunker": v.chunker,
                "chunk_count": len(v.chunks),
                "active_chunk_count": sum(1 for c in v.chunks if c.status == STATUS_ACTIVE),
                "supersedes_id": v.supersedes_id,
                "superseded_by_id": v.superseded_by_id,
            }
            for v in versions
        ],
    )


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentOut]:
    docs = list(session.scalars(select(Document).order_by(Document.created_at.desc())))
    return [_document_out(session, d) for d in docs]


@app.get("/api/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, session: Session = Depends(get_session)) -> DocumentOut:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return _document_out(session, doc)


@app.post("/api/documents/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    category: str = Form("Master Circular"),
    regulator: str = Form("RBI"),
    source_url: str | None = Form(None),
    supersedes: str | None = Form(None),
    chunker: str | None = Form(None),
    session: Session = Depends(get_session),
) -> IngestResponse:
    """Add a new circular. The index updates itself immediately after ingest."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit 25 MB)")

    try:
        result = ingest_upload(
            session,
            data,
            file.filename or "upload.txt",
            title=title,
            source_url=source_url,
            category=category,
            regulator=regulator,
            chunker_name=chunker,
            supersedes_doc_number=supersedes,
        )
    except Exception as exc:
        log.exception("Ingest failed")
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc

    session.commit()
    svc = get_retrieval_service()
    svc.mark_dirty()
    svc.ensure_index()

    detail = (
        f"Already indexed - {result.reason}"
        if result.skipped
        else (
            f"Indexed {result.chunks_created} chunks as version {result.version_number}"
            + (
                f"; {result.chunks_deprecated} chunks from the previous version were marked deprecated"
                if result.chunks_deprecated
                else ""
            )
        )
    )
    return IngestResponse(
        status="skipped" if result.skipped else "indexed",
        detail=detail,
        document=result.as_dict(),
        index=svc.stats(),
    )


@app.post("/api/documents/text", response_model=IngestResponse)
def add_document_text(
    payload: dict,
    session: Session = Depends(get_session),
) -> IngestResponse:
    text = (payload.get("text") or "").strip()
    if len(text) < 200:
        raise HTTPException(400, "Provide at least 200 characters of circular text")
    parsed = build_parsed(
        text,
        fallback_title=payload.get("title") or "Pasted circular",
        source_url=payload.get("source_url"),
    )
    if payload.get("title"):
        parsed.title = payload["title"]
    result = ingest_parsed(
        session,
        parsed,
        category=payload.get("category", "Master Circular"),
        regulator=payload.get("regulator", "RBI"),
        chunker_name=payload.get("chunker"),
        supersedes_doc_number=payload.get("supersedes"),
    )
    session.commit()
    svc = get_retrieval_service()
    svc.mark_dirty()
    svc.ensure_index()
    return IngestResponse(
        status="skipped" if result.skipped else "indexed",
        detail=result.reason or f"Indexed {result.chunks_created} chunks",
        document=result.as_dict(),
        index=svc.stats(),
    )


@app.get("/api/chunks/{chunk_id}")
def get_chunk(chunk_id: str, session: Session = Depends(get_session)) -> dict:
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        raise HTTPException(404, "Chunk not found")
    version = session.get(DocumentVersion, chunk.document_version_id)
    document = session.get(Document, version.document_id) if version else None
    return {
        "chunk_id": chunk.id,
        "text": chunk.chunk_text,
        "chunk_type": chunk.chunk_type,
        "section_ref": chunk.section_ref,
        "status": chunk.status,
        "token_count": chunk.token_count,
        "document": {
            "doc_number": document.doc_number if document else None,
            "title": document.title if document else None,
            "source_url": document.source_url if document else None,
            "version_number": version.version_number if version else None,
            "version_status": version.status if version else None,
            "published_date": version.published_date.isoformat()
            if version and version.published_date
            else None,
        },
    }


# ------------------------------------------------------------------ history
@app.get("/api/sessions", response_model=list[SessionOut])
def list_sessions(limit: int = 30, session: Session = Depends(get_session)) -> list[SessionOut]:
    rows = list(
        session.scalars(select(ChatSession).order_by(ChatSession.created_at.desc()).limit(limit))
    )
    return [
        SessionOut(id=s.id, title=s.title, created_at=s.created_at.isoformat(), messages=[])
        for s in rows
    ]


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
def get_chat_session(session_id: str, session: Session = Depends(get_session)) -> SessionOut:
    s = session.get(ChatSession, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    return SessionOut(
        id=s.id,
        title=s.title,
        created_at=s.created_at.isoformat(),
        messages=[
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "citations": json.loads(m.citations_json or "[]"),
                "created_at": m.created_at.isoformat(),
            }
            for m in s.messages
        ],
    )


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, session: Session = Depends(get_session)) -> dict:
    s = session.get(ChatSession, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    session.delete(s)
    return {"status": "deleted", "session_id": session_id}


# ------------------------------------------------------------------ indexing
@app.post("/api/reindex")
def reindex() -> dict:
    svc = get_retrieval_service()
    svc.mark_dirty()
    with session_scope() as session:
        svc.ensure_index(session)
    return {"status": "ok", "index": svc.stats()}
