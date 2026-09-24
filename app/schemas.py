"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)
    session_id: Optional[str] = None
    mode: Optional[RetrievalMode] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=25)
    include_deprecated: bool = False
    use_llm_judge: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    question: str
    answer: str
    citations: list[dict[str, Any]]
    faithfulness: dict[str, Any]
    retrieval: dict[str, Any]
    usage: dict[str, Any]
    notices: list[str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    mode: Optional[RetrievalMode] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    include_deprecated: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    retrieval: dict[str, Any]


class IngestResponse(BaseModel):
    status: str
    detail: str
    document: dict[str, Any]
    index: dict[str, Any]


class DocumentVersionOut(BaseModel):
    id: str
    version_number: int
    status: str
    published_date: Optional[str]
    effective_date: Optional[str]
    chunker: str
    chunk_count: int
    active_chunk_count: int
    supersedes_id: Optional[str]
    superseded_by_id: Optional[str]


class DocumentOut(BaseModel):
    id: str
    doc_number: str
    title: str
    regulator: str
    category: str
    source_url: Optional[str]
    versions: list[DocumentVersionOut]


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: str


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    messages: list[ChatMessageOut] = []


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
    embedding: str
    reranker: str
    llm: str
    retrieval_mode: str
    chunker: str
    index: dict[str, Any]
