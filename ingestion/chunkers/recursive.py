"""Strategy 2 - recursive/semantic splitting on paragraph then sentence bounds."""
from __future__ import annotations

import re

from ingestion.chunkers.base import TextChunk, count_tokens

_SENT_RE = re.compile(r"(?<=[.;:?!])\s+(?=[A-Z0-9(])")


class RecursiveChunker:
    name = "recursive"

    def __init__(self, chunk_tokens: int = 350, overlap: int = 60) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap

    def _units(self, text: str) -> list[str]:
        units: list[str] = []
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            if count_tokens(para) <= self.chunk_tokens:
                units.append(para)
                continue
            for sent in _SENT_RE.split(para):
                sent = sent.strip()
                if not sent:
                    continue
                if count_tokens(sent) <= self.chunk_tokens:
                    units.append(sent)
                else:  # hard-split pathological sentences
                    words = sent.split()
                    for i in range(0, len(words), self.chunk_tokens):
                        units.append(" ".join(words[i : i + self.chunk_tokens]))
        return units

    def split(self, text: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        buf: list[str] = []
        size = 0
        for unit in self._units(text):
            u_size = count_tokens(unit)
            if size + u_size > self.chunk_tokens and buf:
                chunks.append(TextChunk(text="\n\n".join(buf), ordinal=len(chunks)))
                # carry an overlap tail for context continuity
                tail: list[str] = []
                tail_size = 0
                for prev in reversed(buf):
                    p_size = count_tokens(prev)
                    if tail_size + p_size > self.overlap:
                        break
                    tail.insert(0, prev)
                    tail_size += p_size
                buf, size = list(tail), tail_size
            buf.append(unit)
            size += u_size
        if buf:
            chunks.append(TextChunk(text="\n\n".join(buf), ordinal=len(chunks)))
        return chunks
