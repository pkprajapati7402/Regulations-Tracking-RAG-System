from generation.answer import NO_CONTEXT_ANSWER, extractive_answer, generate_answer
from generation.faithfulness import check_answer, lexical_support
from retrieval.service import get_retrieval_service
from retrieval.types import RetrievedChunk


def _chunk(text, n=1):
    return RetrievedChunk(
        chunk_id=f"c{n}", text=text, score=1.0, doc_number="RBI/2024-25/01",
        doc_title="Test Circular", section_ref="1. Scope",
    )


def test_answer_is_cited_and_faithful(indexed):
    svc = get_retrieval_service()
    chunks, _ = svc.search("What is the cooling off period for digital loans?", top_k=5)
    result = generate_answer("What is the cooling off period for digital loans?", chunks)
    assert "[1]" in result.answer or "[2]" in result.answer
    assert result.faithfulness.rate >= 0.5
    assert result.citations and result.citations[0]["n"] == 1
    assert "not legal or financial advice" in result.answer.lower()


def test_refuses_when_nothing_retrieved():
    result = generate_answer("Who won the 2019 cricket world cup?", [])
    assert NO_CONTEXT_ANSWER in result.answer
    assert result.citations == []


def test_out_of_domain_question_is_not_answered_from_thin_air(indexed):
    svc = get_retrieval_service()
    q = "What is the capital gains tax rate on equity mutual funds in Australia?"
    chunks, _ = svc.search(q, top_k=5)
    result = generate_answer(q, chunks)
    assert NO_CONTEXT_ANSWER in result.answer or result.faithfulness.rate >= 0.5


def test_faithfulness_detects_unsupported_claim():
    sources = [_chunk("Banks shall report incidents within two to six hours of detection.")]
    good = check_answer("Banks must report incidents within two to six hours of detection. [1]", sources)
    bad = check_answer("Banks must report incidents within forty-eight hours to SEBI. [1]", sources)
    assert good.rate == 1.0
    assert bad.rate < good.rate


def test_faithfulness_flags_uncited_and_invalid_citations():
    sources = [_chunk("The cooling off period shall not be less than three days.")]
    report = check_answer(
        "The cooling off period is three days. The lender may also charge a fee. [7]", sources
    )
    assert report.uncited_claims == 1
    assert 7 in report.invalid_citations


def test_lexical_support_rewards_number_agreement():
    passage = "compensation of Rs 100 per day of delay beyond T+5 days"
    assert lexical_support("compensation is Rs 100 per day beyond T+5 days", passage) > 0.6
    assert lexical_support("compensation is Rs 500 per day beyond T+9 days", passage) < 0.6


def test_extractive_answer_cites_every_bullet(indexed):
    svc = get_retrieval_service()
    chunks, _ = svc.search("credit card closure penalty", top_k=4)
    text = extractive_answer("What is the credit card closure penalty?", chunks)
    bullets = [ln for ln in text.split("\n") if ln.startswith("- ")]
    assert bullets
    assert all("[" in b for b in bullets)
