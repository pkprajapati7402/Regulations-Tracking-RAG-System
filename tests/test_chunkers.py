from ingestion.chunkers import get_chunker
from ingestion.chunkers.base import count_tokens
from ingestion.parse import normalise_text

SAMPLE = """RBI/2024-25/99
Master Circular on Testing

1. Introduction
1.1 This paragraph explains the scope of the circular and is deliberately long enough to be a chunk of its own in most strategies, containing several clauses and sub clauses about applicability.

2. Thresholds
2.1 The applicable thresholds are set out in the table below.

[TABLE]
Category | Limit | Timeline
Retail | Rs 50,000 | 7 days
Corporate | Rs 5,00,000 | 15 days
[/TABLE]

3. Reporting
3.1 Entities shall report breaches within seven working days.
"""


def test_all_chunkers_produce_chunks():
    for name in ["fixed", "recursive", "structure_aware"]:
        chunks = get_chunker(name, 120, 20).split(normalise_text(SAMPLE))
        assert chunks, name
        assert all(c.text.strip() for c in chunks)


def test_structure_aware_keeps_table_atomic():
    chunks = get_chunker("structure_aware", 120, 20).split(normalise_text(SAMPLE))
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) == 1
    body = tables[0].text
    assert "Retail | Rs 50,000" in body and "Corporate | Rs 5,00,000" in body


def test_structure_aware_attaches_section_refs():
    chunks = get_chunker("structure_aware", 120, 20).split(normalise_text(SAMPLE))
    refs = [c.section_ref for c in chunks if c.section_ref]
    assert any("Reporting" in r for r in refs)
    assert any(c.text.startswith("[Section:") for c in chunks)


def test_fixed_chunker_respects_budget():
    long_text = " ".join(f"word{i}" for i in range(1000))
    chunks = get_chunker("fixed", 100, 10).split(long_text)
    assert len(chunks) > 5
    assert all(count_tokens(c.text) <= 110 for c in chunks)


def test_recursive_chunker_does_not_explode_budget():
    chunks = get_chunker("recursive", 80, 10).split(normalise_text(SAMPLE))
    assert all(count_tokens(c.text) <= 200 for c in chunks)
