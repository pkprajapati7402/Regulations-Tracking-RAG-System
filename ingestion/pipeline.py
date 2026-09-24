"""Ingestion & re-indexing pipeline.

Responsibilities
----------------
1. Parse a source document (PDF / HTML / text) into normalised text.
2. Chunk it with the configured strategy.
3. Embed the chunks and persist them with full provenance.
4. Handle *amendments*: when a new version of an existing document (or an
   explicit supersession of another document) arrives, insert the new chunks as
   ``active`` and flip the previous version's chunks to ``deprecated`` -- never
   delete -- linking the versions with ``supersedes_id`` / ``superseded_by_id``.

That last step is the stale-data guarantee the whole project is built around.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import settings
from core.logging_conf import get_logger
from core.models import (
    STATUS_ACTIVE,
    STATUS_DEPRECATED,
    Chunk,
    Document,
    DocumentVersion,
)
from ingestion.chunkers import get_chunker
from ingestion.embed import content_hash, get_embedder, pack
from ingestion.parse import ParsedDocument, build_parsed, parse_bytes, parse_file

log = get_logger(__name__)


@dataclass
class IngestResult:
    document_id: str
    version_id: str
    doc_number: str
    title: str
    version_number: int
    chunks_created: int
    chunks_deprecated: int
    superseded_version_id: Optional[str]
    skipped: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "doc_number": self.doc_number,
            "title": self.title,
            "version_number": self.version_number,
            "chunks_created": self.chunks_created,
            "chunks_deprecated": self.chunks_deprecated,
            "superseded_version_id": self.superseded_version_id,
            "skipped": self.skipped,
            "reason": self.reason,
        }


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    return "".join(keep).strip("-")[:80] or "document"


def _find_document(session: Session, doc_number: str) -> Document | None:
    return session.scalar(select(Document).where(Document.doc_number == doc_number))


def resolve_reference(session: Session, reference: str) -> Document | None:
    """Resolve a circular reference to an indexed document.

    RBI documents carry *two* identifiers: the portal number ("RBI/2015-16/18")
    and the departmental circular number ("DBR.AML.BC.No.81/14.01.001/2015-16").
    A supersession clause may quote either, so we try an exact doc_number match
    first and then fall back to searching the header of each indexed version.
    """
    reference = reference.strip()
    if not reference:
        return None
    exact = _find_document(session, reference)
    if exact is not None:
        return exact

    needle = reference.lower()
    for version in session.scalars(select(DocumentVersion)):
        if needle in (version.raw_text or "")[:3000].lower():
            return session.get(Document, version.document_id)
    return None


def _latest_active_version(session: Session, document_id: str) -> DocumentVersion | None:
    return session.scalar(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.status == STATUS_ACTIVE,
        )
        .order_by(DocumentVersion.version_number.desc())
    )


def _chunk_in_scope(chunk: Chunk, scope: list[str]) -> bool:
    """Does this chunk belong to one of the superseded paragraph numbers?"""
    if not scope:
        return True
    ref = (chunk.section_ref or "").strip()
    head = ref.split(">")[0].strip() if ref else ""
    candidates = [head] + [p.strip() for p in ref.split(">")]
    for para in scope:
        for cand in candidates:
            if re.match(rf"^{re.escape(para)}(\.|\s|$|\))", cand):
                return True
        # fall back to the chunk body for chunkers that carry no section_ref
        if re.search(rf"(?m)^{re.escape(para)}[.)]\s", chunk.chunk_text):
            return True
    return False


_TOP_HEADING_RE = re.compile(r"(?m)^(\d{1,2})[.)]\s+\S")


def _split_chunk_by_scope(chunk: Chunk, scope: list[str]) -> tuple[str, str] | None:
    """Split a chunk into (in-scope, out-of-scope) text when it straddles sections.

    A structure-aware chunk can hold several short top-level paragraphs. If an
    amendment supersedes only one of them, deprecating the whole chunk would
    wrongly hide live rules. In that case the chunk is surgically split: the
    superseded part is deprecated and the untouched part is re-inserted as a
    fresh active chunk.
    """
    lines = chunk.chunk_text.split("\n")
    current: str | None = None
    in_scope: list[str] = []
    out_scope: list[str] = []
    seen_headings = False
    for line in lines:
        m = _TOP_HEADING_RE.match(line)
        if m:
            seen_headings = True
            current = m.group(1).split(".")[0]
        target = in_scope if (current in scope) else out_scope
        target.append(line)
    if not seen_headings or not in_scope or not out_scope:
        return None
    if len("\n".join(out_scope).strip()) < 80:
        return None
    return "\n".join(in_scope).strip(), "\n".join(out_scope).strip()


def deprecate_version(
    session: Session,
    version: DocumentVersion,
    superseded_by: DocumentVersion,
    scope: list[str] | None = None,
) -> int:
    """Deprecate a prior version - fully, or only the paragraphs in ``scope``.

    Nothing is ever deleted: deprecated chunks stay in the database, are
    excluded from default retrieval, and remain queryable in audit mode.
    """
    scope = scope or []
    embedder = get_embedder()
    count = 0
    for chunk in list(version.chunks):
        if chunk.status == STATUS_DEPRECATED or not _chunk_in_scope(chunk, scope):
            continue
        if scope:
            split = _split_chunk_by_scope(chunk, scope)
            if split is not None:
                superseded_text, surviving_text = split
                chunk.chunk_text = superseded_text
                surviving_ref = chunk.section_ref
                heading = _TOP_HEADING_RE.search(surviving_text)
                if heading:
                    surviving_ref = surviving_text[heading.start():].split("\n")[0].strip()
                vec = embedder.encode_one(surviving_text)
                session.add(
                    Chunk(
                        document_version_id=version.id,
                        ordinal=chunk.ordinal,
                        chunk_text=surviving_text,
                        chunk_type=chunk.chunk_type,
                        section_ref=surviving_ref,
                        token_count=len(surviving_text.split()),
                        status=STATUS_ACTIVE,
                        embedding=pack(vec),
                        embedding_dim=int(len(vec)),
                    )
                )
                log.info("Split chunk %s: superseded part deprecated, remainder kept active", chunk.id)
        chunk.status = STATUS_DEPRECATED
        count += 1

    remaining_active = sum(1 for c in version.chunks if c.status == STATUS_ACTIVE)
    if remaining_active == 0:
        version.status = STATUS_DEPRECATED
    version.superseded_by_id = superseded_by.id
    session.flush()
    log.info(
        "Deprecated %d chunk(s) of version %s (scope=%s, %d still active); superseded by %s",
        count,
        version.id,
        scope or "whole document",
        remaining_active,
        superseded_by.id,
    )
    return count


def ingest_parsed(
    session: Session,
    parsed: ParsedDocument,
    *,
    category: str = "Master Circular",
    regulator: str = "RBI",
    chunker_name: str | None = None,
    supersedes_doc_number: str | None = None,
    effective_date: date | None = None,
    allow_duplicate: bool = False,
) -> IngestResult:
    chunker_name = chunker_name or settings.chunker
    embedder = get_embedder()

    doc_number = parsed.doc_number or f"LOCAL/{_slug(parsed.title)}"
    sha = content_hash(parsed.text)

    existing_same_content = session.scalar(
        select(DocumentVersion).where(DocumentVersion.content_sha256 == sha)
    )
    if existing_same_content is not None and not allow_duplicate:
        return IngestResult(
            document_id=existing_same_content.document_id,
            version_id=existing_same_content.id,
            doc_number=doc_number,
            title=parsed.title,
            version_number=existing_same_content.version_number,
            chunks_created=0,
            chunks_deprecated=0,
            superseded_version_id=None,
            skipped=True,
            reason="identical content already indexed (sha256 match)",
        )

    document = _find_document(session, doc_number)
    if document is None:
        document = Document(
            doc_number=doc_number,
            title=parsed.title,
            regulator=regulator,
            category=category,
            source_url=parsed.source_url,
        )
        session.add(document)
        session.flush()

    previous = _latest_active_version(session, document.id)
    version_number = (previous.version_number + 1) if previous else 1

    version = DocumentVersion(
        document_id=document.id,
        version_number=version_number,
        published_date=parsed.published_date,
        effective_date=effective_date,
        status=STATUS_ACTIVE,
        source_path=parsed.source_path,
        content_sha256=sha,
        chunker=chunker_name,
        raw_text=parsed.text,
    )
    session.add(version)
    session.flush()

    # --- chunk + embed ---
    chunker = get_chunker(chunker_name, settings.chunk_tokens, settings.chunk_overlap)
    text_chunks = chunker.split(parsed.text)
    vectors = embedder.encode([c.text for c in text_chunks]) if text_chunks else []
    for tc, vec in zip(text_chunks, vectors):
        session.add(
            Chunk(
                document_version_id=version.id,
                ordinal=tc.ordinal,
                chunk_text=tc.text,
                chunk_type=tc.chunk_type,
                section_ref=tc.section_ref,
                token_count=tc.token_count,
                status=STATUS_ACTIVE,
                embedding=pack(vec),
                embedding_dim=int(len(vec)),
            )
        )
    session.flush()

    # --- supersession handling ---
    deprecated = 0
    superseded_id: str | None = None

    if previous is not None:
        deprecated += deprecate_version(session, previous, version)
        version.supersedes_id = previous.id
        superseded_id = previous.id

    explicit_ref = supersedes_doc_number or parsed.supersedes_hint
    if explicit_ref:
        other = resolve_reference(session, explicit_ref)
        if other is not None and other.id != document.id:
            other_version = _latest_active_version(session, other.id)
            if other_version is not None:
                deprecated += deprecate_version(
                    session, other_version, version, scope=parsed.supersedes_scope
                )
                version.supersedes_id = version.supersedes_id or other_version.id
                superseded_id = superseded_id or other_version.id
        elif other is None:
            log.info("Supersession target %s not in corpus yet; recorded on document only.", explicit_ref)

    session.flush()
    log.info(
        "Ingested %s v%d: %d chunks (%s), %d chunks deprecated",
        doc_number,
        version_number,
        len(text_chunks),
        chunker_name,
        deprecated,
    )
    return IngestResult(
        document_id=document.id,
        version_id=version.id,
        doc_number=doc_number,
        title=document.title,
        version_number=version_number,
        chunks_created=len(text_chunks),
        chunks_deprecated=deprecated,
        superseded_version_id=superseded_id,
    )


def ingest_file_path(session: Session, path: str | Path, **kwargs) -> IngestResult:
    parsed = parse_file(path)
    return ingest_parsed(session, parsed, **kwargs)


def ingest_upload(
    session: Session,
    data: bytes,
    filename: str,
    *,
    title: str | None = None,
    source_url: str | None = None,
    **kwargs,
) -> IngestResult:
    """Ingest a user-uploaded circular (the 'add new circular' path in the UI)."""
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / filename
    counter = 1
    while dest.exists():
        dest = settings.upload_dir / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
        counter += 1
    dest.write_bytes(data)

    raw = parse_bytes(data, filename)
    parsed = build_parsed(
        raw, fallback_title=title or Path(filename).stem, source_path=str(dest), source_url=source_url
    )
    if title:
        parsed.title = title
    return ingest_parsed(session, parsed, **kwargs)


def ingest_directory(
    session: Session, directory: str | Path, patterns: Iterable[str] = ("*.txt", "*.md", "*.pdf", "*.html")
) -> list[IngestResult]:
    directory = Path(directory)
    results: list[IngestResult] = []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.rglob(pattern)))
    for path in files:
        if path.name.lower() in {"readme.md", "manifest.json"}:
            continue
        try:
            results.append(ingest_file_path(session, path))
        except Exception as exc:  # pragma: no cover - per-file resilience
            log.exception("Failed to ingest %s: %s", path, exc)
    return results


def reembed_all(session: Session, batch_size: int = 64) -> int:
    """Re-embed every chunk with the currently configured embedder."""
    embedder = get_embedder()
    chunks = list(session.scalars(select(Chunk)))
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vecs = embedder.encode([c.chunk_text for c in batch])
        for chunk, vec in zip(batch, vecs):
            chunk.embedding = pack(vec)
            chunk.embedding_dim = int(len(vec))
    session.flush()
    log.info("Re-embedded %d chunks with %s", len(chunks), embedder.signature)
    return len(chunks)


def rechunk_all(session: Session, chunker_name: str) -> int:
    """Re-chunk + re-embed every active version with a different strategy.

    Used by the chunking-strategy comparison in the evaluation harness.

    Note: chunk-level deprecation flags cannot survive a re-chunk (chunk
    boundaries change), so chunks inherit their version's status. Re-run
    ``python -m ingestion.reindex --reset --seed`` afterwards to restore the
    partial-supersession state used by the amendment case study.
    """
    embedder = get_embedder()
    chunker = get_chunker(chunker_name, settings.chunk_tokens, settings.chunk_overlap)
    versions = list(session.scalars(select(DocumentVersion)))
    total = 0
    for version in versions:
        for chunk in list(version.chunks):
            session.delete(chunk)
        session.flush()
        pieces = chunker.split(version.raw_text)
        vecs = embedder.encode([p.text for p in pieces]) if pieces else []
        for piece, vec in zip(pieces, vecs):
            session.add(
                Chunk(
                    document_version_id=version.id,
                    ordinal=piece.ordinal,
                    chunk_text=piece.text,
                    chunk_type=piece.chunk_type,
                    section_ref=piece.section_ref,
                    token_count=piece.token_count,
                    status=version.status,
                    embedding=pack(vec),
                    embedding_dim=int(len(vec)),
                )
            )
        version.chunker = chunker_name
        total += len(pieces)
    session.flush()
    log.info("Re-chunked %d versions into %d chunks using %s", len(versions), total, chunker_name)
    return total
