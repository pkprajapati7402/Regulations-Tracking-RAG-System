"""CLI for building / rebuilding the index.

    python -m ingestion.reindex --seed                 # index corpus/seed
    python -m ingestion.reindex --path corpus/raw      # index downloaded RBI docs
    python -m ingestion.reindex --file new-circular.pdf --supersedes "RBI/2015-16/18"
    python -m ingestion.reindex --rechunk fixed        # re-chunk everything (eval sweeps)
    python -m ingestion.reindex --reembed              # re-embed after switching provider
    python -m ingestion.reindex --status               # show what is indexed
    python -m ingestion.reindex --reset --seed         # wipe and rebuild
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import select

from core.config import settings
from core.db import get_engine, init_db, session_scope
from core.logging_conf import get_logger
from core.models import STATUS_ACTIVE, Base, Chunk, Document, DocumentVersion
from ingestion.pipeline import (
    ingest_directory,
    ingest_file_path,
    rechunk_all,
    reembed_all,
)
from retrieval.service import get_retrieval_service

log = get_logger(__name__)


def reset_database() -> None:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    log.info("Database reset.")


def print_status() -> None:
    with session_scope() as session:
        docs = list(session.scalars(select(Document)))
        print(f"\nIndexed documents: {len(docs)}")
        print("-" * 96)
        for doc in docs:
            versions = list(
                session.scalars(
                    select(DocumentVersion)
                    .where(DocumentVersion.document_id == doc.id)
                    .order_by(DocumentVersion.version_number)
                )
            )
            print(f"{doc.doc_number}\n  {doc.title}")
            for v in versions:
                n_chunks = session.scalar(
                    select(Chunk).where(Chunk.document_version_id == v.id).with_only_columns(Chunk.id).limit(1)
                )
                total = len(v.chunks)
                active = sum(1 for c in v.chunks if c.status == STATUS_ACTIVE)
                flag = "ACTIVE   " if v.status == STATUS_ACTIVE else "SUPERSEDED"
                print(
                    f"    v{v.version_number} [{flag}] published={v.published_date} "
                    f"chunker={v.chunker} chunks={total} (active={active})"
                    + (f" superseded_by={v.superseded_by_id[:8]}" if v.superseded_by_id else "")
                )
        svc = get_retrieval_service()
        svc.mark_dirty()
        print("-" * 96)
        print("Retrieval index:", svc.stats(), "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the regulatory index.")
    parser.add_argument("--seed", action="store_true", help="Ingest corpus/seed")
    parser.add_argument("--path", help="Ingest every supported file in a directory")
    parser.add_argument("--file", help="Ingest a single file")
    parser.add_argument("--supersedes", help="doc_number that the --file document supersedes")
    parser.add_argument("--chunker", default=None, help="Override the chunker for this ingest")
    parser.add_argument("--rechunk", help="Re-chunk the whole corpus with the named strategy")
    parser.add_argument("--reembed", action="store_true", help="Re-embed all chunks")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables first")
    parser.add_argument("--status", action="store_true", help="Print index status and exit")
    args = parser.parse_args()

    init_db()
    if args.reset:
        reset_database()

    if args.status and not any([args.seed, args.path, args.file, args.rechunk, args.reembed]):
        print_status()
        return

    with session_scope() as session:
        if args.seed:
            results = ingest_directory(session, settings.corpus_dir / "seed")
            for r in results:
                print(
                    f"  {'skipped ' if r.skipped else 'indexed '} {r.doc_number} v{r.version_number} "
                    f"chunks={r.chunks_created} deprecated={r.chunks_deprecated} {r.reason}"
                )
        if args.path:
            results = ingest_directory(session, Path(args.path))
            for r in results:
                print(
                    f"  {'skipped ' if r.skipped else 'indexed '} {r.doc_number} v{r.version_number} "
                    f"chunks={r.chunks_created} deprecated={r.chunks_deprecated} {r.reason}"
                )
        if args.file:
            r = ingest_file_path(
                session,
                args.file,
                chunker_name=args.chunker,
                supersedes_doc_number=args.supersedes,
            )
            print(f"  indexed {r.doc_number} v{r.version_number} chunks={r.chunks_created} "
                  f"deprecated={r.chunks_deprecated}")
        if args.rechunk:
            total = rechunk_all(session, args.rechunk)
            print(f"  re-chunked corpus into {total} chunks using '{args.rechunk}'")
        if args.reembed:
            total = reembed_all(session)
            print(f"  re-embedded {total} chunks")

    get_retrieval_service().mark_dirty()
    print_status()


if __name__ == "__main__":  # pragma: no cover
    main()
