"""Reproduce the stale-data / amendment case study end to end.

    python scripts/case_study_amendment.py

Steps
-----
1. Reset the index and ingest ONLY the 2016 KYC Master Direction.
2. Ask "how often must high risk customers' KYC be updated?" -> the 2016 rule
   (two years) is retrieved and answered.
3. Push the 2026 amendment through the ingestion pipeline. It declares
   "in supersession of ... DBR.AML.BC.No.81/14.01.001/2015-16", so the pipeline
   deprecates the superseded version's chunks.
4. Re-run the identical query -> the answer now reflects the amended rule
   (three years) and the superseded chunk no longer appears by default.
5. Re-run with include_deprecated=True -> the superseded text is still there,
   for audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings  # noqa: E402
from core.db import init_db, session_scope  # noqa: E402
from core.models import Base  # noqa: E402
from core.db import get_engine  # noqa: E402
from generation.answer import generate_answer  # noqa: E402
from ingestion.pipeline import ingest_file_path  # noqa: E402
from retrieval.service import get_retrieval_service  # noqa: E402

QUERY = "How often must banks carry out periodic KYC updation for high risk customers?"
SEED = settings.corpus_dir / "seed"


def show(title: str, include_deprecated: bool = False) -> None:
    svc = get_retrieval_service()
    svc.mark_dirty()
    chunks, debug = svc.search(QUERY, top_k=4, include_deprecated=include_deprecated)
    answer = generate_answer(QUERY, chunks)
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"index: {svc.stats()}  |  mode={debug.mode}  include_deprecated={include_deprecated}")
    print("\nTop passages:")
    for i, c in enumerate(chunks, 1):
        marker = "ACTIVE" if c.status == "active" else "SUPERSEDED"
        line = " ".join(c.text.split())[:150]
        print(f"  [{i}] {marker:<10} {c.doc_number:<28} {line}…")
    print("\nAnswer:\n" + answer.answer)
    print(f"\nfaithfulness={answer.faithfulness.rate:.2f}  notices={answer.notices}")


def main() -> None:
    init_db()
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    get_retrieval_service().mark_dirty()

    with session_scope() as s:
        r = ingest_file_path(s, SEED / "01-kyc-master-direction-2016.txt")
        print(f"Indexed base document {r.doc_number} -> {r.chunks_created} chunks")
    show("STEP 1 — BEFORE the amendment (only the 2016 Master Direction is indexed)")

    with session_scope() as s:
        r = ingest_file_path(s, SEED / "02-kyc-amendment-2026.txt")
        print(
            f"\nIngested amendment {r.doc_number}: +{r.chunks_created} chunks, "
            f"{r.chunks_deprecated} chunks deprecated, supersedes={r.superseded_version_id}"
        )
    show("STEP 2 — AFTER the amendment (default retrieval: superseded text excluded)")
    show("STEP 3 — AUDIT MODE (superseded text deliberately included)", include_deprecated=True)


if __name__ == "__main__":
    main()
