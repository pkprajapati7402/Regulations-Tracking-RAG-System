"""Strategy 1 - fixed-size token windows with overlap (the naive baseline)."""
from __future__ import annotations

from ingestion.chunkers.base import TextChunk, count_tokens


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, chunk_tokens: int = 350, overlap: int = 60) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap = min(overlap, chunk_tokens - 1)

    def split(self, text: str) -> list[TextChunk]:
        words = text.split()
        if not words:
            return []
        step = self.chunk_tokens - self.overlap
        chunks: list[TextChunk] = []
        for i in range(0, len(words), step):
            window = words[i : i + self.chunk_tokens]
            if not window:
                break
            body = " ".join(window)
            if count_tokens(body) < 15 and chunks:
                break
            chunks.append(TextChunk(text=body, chunk_type="prose", ordinal=len(chunks)))
            if i + self.chunk_tokens >= len(words):
                break
        return chunks
