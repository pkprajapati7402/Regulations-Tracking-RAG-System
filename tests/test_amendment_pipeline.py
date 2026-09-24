"""The stale-data guarantee: an amendment must deprecate, not delete."""
from pathlib import Path

from sqlalchemy import select

from core.db import init_db, reset_engine, session_scope
from core.models import STATUS_ACTIVE, STATUS_DEPRECATED, Base, Chunk, Document, DocumentVersion
from core.db import get_engine
from ingestion.pipeline import ingest_file_path
from retrieval.service import get_retrieval_service, reset_retrieval_service

SEED = Path(__file__).resolve().parent.parent / "corpus" / "seed"


def _fresh_db():
    reset_engine()
    reset_retrieval_service()
    init_db()
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def test_amendment_deprecates_superseded_version_and_updates_answers(_env):
    _fresh_db()
    with session_scope() as s:
        base = ingest_file_path(s, SEED / "01-kyc-master-direction-2016.txt")
    assert base.chunks_created > 0

    svc = get_retrieval_service()
    svc.mark_dirty()
    before, _ = svc.search("periodic updation of KYC for high risk customers", top_k=5)
    assert any("two years" in r.text for r in before)

    with session_scope() as s:
        amended = ingest_file_path(s, SEED / "02-kyc-amendment-2026.txt")
    assert amended.chunks_deprecated > 0, "the superseded version's chunks must be deprecated"
    assert amended.superseded_version_id

    svc.mark_dirty()
    after, _ = svc.search("periodic updation of KYC for high risk customers", top_k=5)
    assert any("three years" in r.text for r in after), "amended rule must surface"
    assert all(r.status == STATUS_ACTIVE for r in after)
    assert not any("once in every two years" in r.text for r in after), "superseded rule must not surface"

    # nothing was deleted - the old text is still auditable
    audit, _ = svc.search(
        "periodic updation of KYC for high risk customers", top_k=10, include_deprecated=True
    )
    assert any(r.status == STATUS_DEPRECATED for r in audit)

    with session_scope() as s:
        old = s.scalar(
            select(DocumentVersion).where(DocumentVersion.superseded_by_id.is_not(None))
        )
        assert old is not None
        # The amendment is *scoped* to paragraph 6, so only those chunks are
        # deprecated - the rest of the Master Direction stays live.
        deprecated = [c for c in old.chunks if c.status == STATUS_DEPRECATED]
        still_active = [c for c in old.chunks if c.status == STATUS_ACTIVE]
        assert deprecated and still_active
        assert all("Periodic Updation" in (c.section_ref or "") for c in deprecated)
        assert s.scalar(select(Chunk).where(Chunk.status == STATUS_DEPRECATED)) is not None


def test_unscoped_supersession_deprecates_the_whole_version(_env):
    """Without a paragraph scope, the entire prior version is superseded."""
    _fresh_db()
    from ingestion.parse import build_parsed
    from ingestion.pipeline import ingest_parsed

    base = (SEED / "08-cyber-security-framework-banks.txt").read_text()
    with session_scope() as s:
        ingest_parsed(s, build_parsed(base, "cyber"))
        result = ingest_parsed(
            s,
            build_parsed(
                "RBI/2027-28/01\nRevised Cyber Security Framework in Banks\n\n01 April 2027\n\n"
                "This circular is issued in supersession of DBS.CO/CSITE/BC.11/33.01.001/2015-16.\n\n"
                "1. Reporting\n1.1 Banks shall report all cyber incidents within one hour of detection.\n",
                "revised",
            ),
        )
    assert result.chunks_deprecated > 0
    with session_scope() as s:
        old = s.scalar(select(DocumentVersion).where(DocumentVersion.status == STATUS_DEPRECATED))
        assert old is not None and old.superseded_by_id is not None
        assert all(c.status == STATUS_DEPRECATED for c in old.chunks)


def test_reingesting_same_document_is_idempotent(_env):
    _fresh_db()
    with session_scope() as s:
        first = ingest_file_path(s, SEED / "03-digital-lending-guidelines.txt")
        second = ingest_file_path(s, SEED / "03-digital-lending-guidelines.txt")
    assert not first.skipped
    assert second.skipped and "sha256" in second.reason
    with session_scope() as s:
        assert len(list(s.scalars(select(Document)))) == 1


def test_new_version_of_same_doc_number_bumps_version(_env):
    _fresh_db()
    text = (SEED / "05-master-circular-customer-service.txt").read_text()
    from ingestion.parse import build_parsed
    from ingestion.pipeline import ingest_parsed

    with session_scope() as s:
        v1 = ingest_parsed(s, build_parsed(text, "cs"))
        v2 = ingest_parsed(s, build_parsed(text + "\n\n8. New paragraph added in the revision.", "cs"))
    assert v1.version_number == 1 and v2.version_number == 2
    assert v2.chunks_deprecated > 0
