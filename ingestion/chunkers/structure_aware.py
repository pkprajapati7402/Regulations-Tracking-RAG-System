"""Strategy 3 - structure-aware chunking for regulatory circulars.

Rules:
  * Split on headings / numbered paragraph markers ("4.", "4.2", "Annex II",
    "Chapter III", ALL-CAPS headings).
  * Keep tables atomic: a [TABLE]...[/TABLE] block (or a run of pipe-delimited
    lines) is never split, and carries chunk_type='table'.
  * Attach the section path ("4. Customer Due Diligence > 4.2 Periodic Updation")
    as metadata and prepend it to the chunk text so the embedding sees context.
  * Oversized sections are split on paragraph boundaries but keep their
    section_ref, so provenance survives.
"""
from __future__ import annotations

import re

from ingestion.chunkers.base import TextChunk, count_tokens

_NUMBERED_RE = re.compile(r"^((?:\d{1,2})(?:\.\d{1,2}){0,3})[.)]?\s+(\S.{0,180})$")
_ANNEX_RE = re.compile(r"^((?:Annex(?:ure)?|Appendix|Schedule|Chapter|Part|Section)\b[^\n]{0,120})$", re.I)
_CAPS_RE = re.compile(r"^([A-Z][A-Z0-9 ,&()\-/'\.]{6,90})$")
_TABLE_OPEN = "[TABLE]"
_TABLE_CLOSE = "[/TABLE]"


def _heading_level(line: str) -> tuple[int, str] | None:
    line = line.strip()
    if not line or len(line) > 200:
        return None
    m = _NUMBERED_RE.match(line)
    if m:
        return len(m.group(1).split(".")), line
    if _ANNEX_RE.match(line):
        return 1, line
    if _CAPS_RE.match(line) and not line.endswith((".", ",")):
        return 1, line
    return None


class _Block:
    __slots__ = ("kind", "lines", "section_path")

    def __init__(self, kind: str, section_path: list[str]):
        self.kind = kind
        self.lines: list[str] = []
        self.section_path = list(section_path)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


class StructureAwareChunker:
    name = "structure_aware"

    def __init__(self, chunk_tokens: int = 350, overlap: int = 60, min_tokens: int = 40) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self.min_tokens = min_tokens

    # -- pass 1: segment the document into section/table blocks --------------
    def _blocks(self, text: str) -> list[_Block]:
        blocks: list[_Block] = []
        path: list[str] = []
        current = _Block("prose", path)
        in_table = False

        for raw_line in text.split("\n"):
            line = raw_line.rstrip()

            if _TABLE_OPEN in line:
                if current.text:
                    blocks.append(current)
                current = _Block("table", path)
                in_table = True
                continue
            if _TABLE_CLOSE in line:
                if current.text:
                    blocks.append(current)
                current = _Block("prose", path)
                in_table = False
                continue
            if in_table:
                current.lines.append(line)
                continue

            # A run of pipe-delimited lines is also treated as a table.
            if line.count("|") >= 2:
                if current.kind != "table":
                    if current.text:
                        blocks.append(current)
                    current = _Block("table", path)
                current.lines.append(line)
                continue
            if current.kind == "table" and line.strip():
                blocks.append(current)
                current = _Block("prose", path)

            heading = _heading_level(line)
            if heading:
                level, title = heading
                if current.text:
                    blocks.append(current)
                path = path[: level - 1]
                path.append(title.strip())
                current = _Block("prose", path)
                current.lines.append(line)
                continue

            current.lines.append(line)

        if current.text:
            blocks.append(current)
        return [b for b in blocks if b.text]

    # -- pass 2: pack blocks into chunks respecting the token budget ---------
    def split(self, text: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        buffer: list[_Block] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buffer, buf_tokens
            if not buffer:
                return
            section = buffer[0].section_path
            body = "\n\n".join(b.text for b in buffer)
            chunks.append(self._make(body, "prose", section, len(chunks)))
            buffer, buf_tokens = [], 0

        for block in self._blocks(text):
            tokens = count_tokens(block.text)

            if block.kind == "table":
                flush()
                chunks.append(self._make(block.text, "table", block.section_path, len(chunks)))
                continue

            if tokens > self.chunk_tokens:
                flush()
                for piece in self._split_long(block.text):
                    chunks.append(self._make(piece, "prose", block.section_path, len(chunks)))
                continue

            if buf_tokens + tokens > self.chunk_tokens:
                flush()
            buffer.append(block)
            buf_tokens += tokens

        flush()
        return self._merge_tiny(chunks)

    def _split_long(self, text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        out: list[str] = []
        buf: list[str] = []
        size = 0
        for para in paras:
            p_size = count_tokens(para)
            if p_size > self.chunk_tokens:
                if buf:
                    out.append("\n\n".join(buf))
                    buf, size = [], 0
                words = para.split()
                step = self.chunk_tokens - self.overlap
                for i in range(0, len(words), step):
                    out.append(" ".join(words[i : i + self.chunk_tokens]))
                continue
            if size + p_size > self.chunk_tokens and buf:
                out.append("\n\n".join(buf))
                buf, size = [], 0
            buf.append(para)
            size += p_size
        if buf:
            out.append("\n\n".join(buf))
        return out

    def _make(self, body: str, kind: str, path: list[str], ordinal: int) -> TextChunk:
        section_ref = " > ".join(path) if path else None
        prefixed = f"[Section: {section_ref}]\n{body}" if section_ref else body
        return TextChunk(
            text=prefixed,
            chunk_type=kind,
            section_ref=section_ref,
            ordinal=ordinal,
            meta={"raw_text": body},
        )

    def _merge_tiny(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Fold chunks below the minimum size into their neighbour (tables excluded)."""
        merged: list[TextChunk] = []
        for chunk in chunks:
            if (
                merged
                and chunk.chunk_type == "prose"
                and merged[-1].chunk_type == "prose"
                and count_tokens(chunk.text) < self.min_tokens
            ):
                merged[-1].text = merged[-1].text + "\n\n" + chunk.text
                continue
            chunk.ordinal = len(merged)
            merged.append(chunk)
        return merged
