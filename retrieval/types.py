"""Shared retrieval data types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_number: str
    doc_title: str
    section_ref: str | None = None
    chunk_type: str = "prose"
    status: str = "active"
    version_number: int = 1
    published_date: str | None = None
    source_url: str | None = None
    retriever: str = ""
    components: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(float(self.score), 6),
            "doc_number": self.doc_number,
            "doc_title": self.doc_title,
            "section_ref": self.section_ref,
            "chunk_type": self.chunk_type,
            "status": self.status,
            "version_number": self.version_number,
            "published_date": self.published_date,
            "source_url": self.source_url,
            "retriever": self.retriever,
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
        }
