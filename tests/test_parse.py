from pathlib import Path

from ingestion.parse import build_parsed, extract_date, extract_doc_number, extract_supersedes_hint, parse_file

SEED = Path(__file__).resolve().parent.parent / "corpus" / "seed"


def test_extracts_doc_number_and_date():
    parsed = parse_file(SEED / "02-kyc-amendment-2026.txt")
    assert parsed.doc_number == "RBI/2026-27/88"
    assert parsed.published_date is not None and parsed.published_date.year == 2026
    assert "Amendment Directions" in parsed.title


def test_detects_supersession_language():
    text = (SEED / "02-kyc-amendment-2026.txt").read_text()
    assert extract_supersedes_hint(text) == "DBR.AML.BC.No.81/14.01.001/2015-16"


def test_html_tables_become_pipe_rows():
    html = "<html><body><h1>Circular</h1><table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table></body></html>"
    from ingestion.parse import parse_html

    text = parse_html(html)
    assert "A | B" in text and "1 | 2" in text and "[TABLE]" in text


def test_build_parsed_normalises_unicode():
    parsed = build_parsed("RBI/2024-25/01\nSome \u201cTitle\u201d Here For Testing\n\nBody text.", "fallback")
    assert '"Title"' in parsed.text
    assert extract_doc_number(parsed.text) == "RBI/2024-25/01"
    assert extract_date("published on 12 March 2020") is not None
