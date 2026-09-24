"""Document parsing: PDF / HTML / text -> normalised plain text + metadata.

The parser deliberately keeps structural signals (headings, numbered paragraphs,
pipe-delimited tables) intact in the text, because the structure-aware chunker
downstream depends on them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from core.logging_conf import get_logger

log = get_logger(__name__)

_DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.I), "dmy"),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", re.I), "mdy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
]

_MONTHS = {
    m.lower(): i + 1
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
}

# e.g. "RBI/2024-25/13 DOR.AML.REC.12/14.01.001/2024-25"
_DOC_NUMBER_RE = re.compile(r"\bRBI/(?:[A-Za-z]{2,8}/)?\d{4}-\d{2,4}/\d+\b")
_CIRCULAR_REF_RE = re.compile(
    r"\b[A-Z]{2,6}(?:[\.\(][A-Za-z0-9]{1,12}\)?){1,6}/\d{2}\.\d{2}\.\d{3}/\d{4}-\d{2,4}\b"
)


@dataclass
class ParsedDocument:
    text: str
    title: str
    doc_number: Optional[str] = None
    published_date: Optional[date] = None
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    supersedes_hint: Optional[str] = None
    supersedes_scope: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def normalise_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def extract_date(text: str) -> Optional[date]:
    head = text[:4000]
    for pattern, kind in _DATE_PATTERNS:
        m = pattern.search(head)
        if not m:
            continue
        try:
            if kind == "dmy":
                return date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
            if kind == "mdy":
                return date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except (ValueError, KeyError):  # pragma: no cover - defensive
            continue
    return None


def extract_doc_number(text: str) -> Optional[str]:
    head = text[:4000]
    m = _DOC_NUMBER_RE.search(head)
    if m:
        return m.group(0)
    m = _CIRCULAR_REF_RE.search(head)
    return m.group(0) if m else None


def extract_title(text: str, fallback: str) -> str:
    for line in text.split("\n"):
        line = line.strip(" .:-")
        if len(line) < 12 or len(line) > 220:
            continue
        low = line.lower()
        if low.startswith(("dear", "madam", "sir", "to,", "rbi/", "all ")):
            continue
        if _DOC_NUMBER_RE.search(line) or _CIRCULAR_REF_RE.search(line):
            continue
        if sum(ch.isdigit() for ch in line) > len(line) / 3:
            continue
        return line
    return fallback


def extract_supersedes_hint(text: str) -> Optional[str]:
    """Find an explicit 'supersedes / in supersession of <doc no>' reference."""
    pattern = re.compile(
        r"(?:in\s+supersession\s+of|supersed(?:es|ing)|replaces|amends?\s+the)[^\n]{0,220}",
        re.I,
    )
    for m in pattern.finditer(text):
        snippet = m.group(0)
        ref = _DOC_NUMBER_RE.search(snippet) or _CIRCULAR_REF_RE.search(snippet)
        if ref:
            return ref.group(0)
    return None


_SCOPE_RE = re.compile(
    r"(?:in\s+supersession\s+of|supersed(?:es|ing)|amends?)\s+"
    r"(?:the\s+)?(?:provisions\s+of\s+)?"
    r"(?:paragraphs?|paras?|clauses?|sections?)\s+"
    r"((?:\d+(?:\.\d+)*)(?:\s*(?:,|and|to)\s*\d+(?:\.\d+)*)*)",
    re.I,
)


def extract_supersession_scope(text: str) -> list[str]:
    """Paragraph numbers an amendment explicitly supersedes.

    "in supersession of paragraph 6 of Master Direction X" -> ["6"].
    An empty list means the supersession is not scoped, i.e. the whole prior
    version is replaced. This is what allows a partial amendment to deprecate
    only the paragraphs it actually touches, leaving the rest of the Master
    Direction live and retrievable.
    """
    scope: list[str] = []
    for m in _SCOPE_RE.finditer(text):
        for num in re.findall(r"\d+(?:\.\d+)*", m.group(1)):
            if num not in scope:
                scope.append(num)
    return scope


def parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - malformed pdf
            log.warning("pdf page extraction failed for %s: %s", path, exc)
    return "\n\n".join(pages)


def parse_html(raw_html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Render tables as pipe-delimited rows so the chunker can keep them atomic.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with("\n\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n\n")

    return soup.get_text("\n")


def parse_bytes(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    decoded = data.decode("utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return parse_html(decoded)
    return decoded


def parse_file(path: str | Path) -> ParsedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = parse_pdf(path)
    elif suffix in {".html", ".htm"}:
        raw = parse_html(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    return build_parsed(raw, fallback_title=path.stem, source_path=str(path))


def build_parsed(
    raw_text: str,
    fallback_title: str,
    source_path: str | None = None,
    source_url: str | None = None,
) -> ParsedDocument:
    text = normalise_text(raw_text)
    return ParsedDocument(
        text=text,
        title=extract_title(text, fallback_title),
        doc_number=extract_doc_number(text),
        published_date=extract_date(text),
        source_path=source_path,
        source_url=source_url,
        supersedes_hint=extract_supersedes_hint(text),
        supersedes_scope=extract_supersession_scope(text),
    )
