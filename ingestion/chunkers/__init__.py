"""Chunking strategies, selectable by name so they can be compared head-to-head."""
from __future__ import annotations

from ingestion.chunkers.base import Chunker, TextChunk, count_tokens
from ingestion.chunkers.fixed_size import FixedSizeChunker
from ingestion.chunkers.recursive import RecursiveChunker
from ingestion.chunkers.structure_aware import StructureAwareChunker

CHUNKERS = {
    FixedSizeChunker.name: FixedSizeChunker,
    RecursiveChunker.name: RecursiveChunker,
    StructureAwareChunker.name: StructureAwareChunker,
}


def get_chunker(name: str, chunk_tokens: int = 350, overlap: int = 60) -> Chunker:
    try:
        cls = CHUNKERS[name]
    except KeyError as exc:  # pragma: no cover - config error
        raise ValueError(f"Unknown chunker '{name}'. Options: {sorted(CHUNKERS)}") from exc
    return cls(chunk_tokens=chunk_tokens, overlap=overlap)


__all__ = [
    "Chunker",
    "TextChunk",
    "count_tokens",
    "CHUNKERS",
    "get_chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "StructureAwareChunker",
]
