"""SQLAlchemy data model.

Mirrors the ER diagram in Project-Details.md:

    DOCUMENTS 1--* DOCUMENT_VERSIONS 1--* CHUNKS
    DOCUMENT_VERSIONS *--? DOCUMENT_VERSIONS  (superseded_by)
    EVAL_QUERIES 1--* EVAL_JUDGMENTS *--1 CHUNKS
    EVAL_RUNS 1--* EVAL_RESULTS

Key production detail: nothing is ever hard-deleted on amendment. Superseded
document versions and their chunks move to status='deprecated' so that they are
excluded from default retrieval but remain auditable / queryable on request.
"""
from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


STATUS_ACTIVE = "active"
STATUS_DEPRECATED = "deprecated"


class Document(Base):
    """A logical regulatory document (stable across amendments)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    doc_number: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    regulator: Mapped[str] = mapped_column(String(64), default="RBI")
    category: Mapped[str] = mapped_column(String(128), default="Master Circular")
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentVersion.version_number"
    )


class DocumentVersion(Base):
    """One concrete issued version of a document."""

    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    published_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    effective_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_ACTIVE, index=True)
    supersedes_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True
    )
    superseded_by_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True
    )
    source_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    chunker: Mapped[str] = mapped_column(String(32), default="structure_aware")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A retrievable unit of text plus its dense embedding and provenance."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    chunk_text: Mapped[str] = mapped_column(Text)
    chunk_type: Mapped[str] = mapped_column(String(32), default="prose")  # prose | table | heading
    section_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_ACTIVE, index=True)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    version: Mapped[DocumentVersion] = relationship(back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(Text, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[str] = mapped_column(Text, default="[]")
    meta_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


# ----------------------------- evaluation ---------------------------------


class EvalQuery(Base):
    __tablename__ = "eval_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    query_text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64), default="domain_track")


class EvalJudgment(Base):
    __tablename__ = "eval_judgments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    query_id: Mapped[str] = mapped_column(ForeignKey("eval_queries.id", ondelete="CASCADE"))
    chunk_id: Mapped[str] = mapped_column(String(36))
    relevance_grade: Mapped[int] = mapped_column(Integer, default=0)  # 0 | 1 | 2


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    config_name: Mapped[str] = mapped_column(String(128))
    track: Mapped[str] = mapped_column(String(32), default="domain_track")
    run_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    notes: Mapped[str] = mapped_column(Text, default="")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    eval_run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"))
    query_id: Mapped[str] = mapped_column(String(36))
    precision_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    recall_at_k: Mapped[float] = mapped_column(Float, default=0.0)
    ndcg_at_10: Mapped[float] = mapped_column(Float, default=0.0)
    mrr: Mapped[float] = mapped_column(Float, default=0.0)
