"""Chunker interface shared by the three strategies we compare."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TextChunk:
    text: str
    chunk_type: str = "prose"  # prose | table | heading
    section_ref: str | None = None
    ordinal: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Cheap whitespace/punctuation tokeniser.

    We intentionally avoid a model-specific tokenizer: chunk budgets only need
    to be *consistent* across strategies for the comparison to be fair, and
    this keeps the ingestion path dependency-free.
    """
    return len(_TOKEN_RE.findall(text))


def truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


class Chunker(Protocol):
    name: str

    def split(self, text: str) -> list[TextChunk]:  # pragma: no cover - protocol
        ...
