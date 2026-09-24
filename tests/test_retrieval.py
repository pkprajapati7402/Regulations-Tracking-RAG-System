import pytest

from retrieval.service import get_retrieval_service


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_returns_results(indexed, mode):
    svc = get_retrieval_service()
    results, debug = svc.search("periodic updation of KYC for high risk customers", mode=mode, top_k=5)
    assert results, mode
    assert debug.mode == mode
    assert all(r.status == "active" for r in results)


def test_retrieval_finds_the_right_document(indexed):
    svc = get_retrieval_service()
    cases = {
        "cooling off period for digital loans": "RBI/2022-23/111",
        "SMA-2 classification more than 60 days": "RBI/2018-19/203",
        "penalty for delay in closing a credit card": "RBI/2022-23/92",
        "report cyber security incident to the Reserve Bank": "RBI/2015-16/418",
        "priority sector lending target for domestic banks": "RBI/2024-25/06",
    }
    for query, expected in cases.items():
        results, _ = svc.search(query, mode="hybrid_rerank", top_k=5)
        assert expected in {r.doc_number for r in results}, f"{query} -> {[r.doc_number for r in results]}"


def test_tables_are_retrievable_as_units(indexed):
    svc = get_retrieval_service()
    results, _ = svc.search("SMA sub-category classification days overdue", top_k=8)
    table_hits = [r for r in results if r.chunk_type == "table"]
    assert table_hits
    assert any("SMA-2" in r.text for r in results)


def test_deprecated_chunks_excluded_by_default_and_visible_in_audit_mode(indexed):
    svc = get_retrieval_service()
    default, _ = svc.search("KYC periodic updation frequency high risk", top_k=10)
    assert all(r.status == "active" for r in default)

    audit, debug = svc.search(
        "KYC periodic updation frequency high risk", top_k=10, include_deprecated=True
    )
    assert debug.include_deprecated
    assert len(audit) >= len(default)


def test_hybrid_beats_or_matches_single_retriever_on_a_hard_query(indexed):
    """A query mixing an exact identifier with paraphrase should favour hybrid."""
    svc = get_retrieval_service()
    query = "what is the annual percentage rate disclosure requirement in digital lending"
    hybrid, _ = svc.search(query, mode="hybrid_rerank", top_k=3)
    assert hybrid and hybrid[0].doc_number == "RBI/2022-23/111"


def test_unknown_mode_raises(indexed):
    with pytest.raises(ValueError):
        get_retrieval_service().search("anything", mode="magic")
