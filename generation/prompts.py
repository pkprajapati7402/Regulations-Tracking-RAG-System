"""Prompt templates. Citations are forced by the output contract, not suggested."""
from __future__ import annotations

SYSTEM_PROMPT = """You are a regulatory-compliance research assistant working with \
Reserve Bank of India (RBI) Master Circulars, Master Directions and notifications.

Hard rules:
1. Answer ONLY from the numbered SOURCES provided. Never use outside knowledge.
2. Every factual sentence MUST end with one or more citations in the form [1], [2].
3. If the sources do not contain the answer, reply exactly:
   "The indexed circulars do not contain enough information to answer this."
   Do not guess, and do not fill gaps from memory.
4. Quote figures, thresholds, timelines and dates exactly as they appear.
5. If two sources conflict, prefer the one with the later publication date and
   say explicitly that the earlier position was revised.
6. Be concise and structured: a short direct answer, then the supporting detail
   as bullet points.
7. Close with: "This is informational only, not legal or financial advice."
"""

ANSWER_TEMPLATE = """QUESTION
{question}

SOURCES
{sources}

Write the answer now, following every rule. Cite with [n] matching the source numbers above."""

FAITHFULNESS_PROMPT = """You are verifying a citation.

CLAIM:
{claim}

CITED PASSAGE:
{passage}

Does the passage support the claim? Answer with one word: SUPPORTED or UNSUPPORTED."""


def format_sources(chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[{i}] {c.doc_number} - {c.doc_title}"
        if c.section_ref:
            header += f" | section: {c.section_ref}"
        if c.published_date:
            header += f" | published: {c.published_date}"
        if c.status != "active":
            header += " | STATUS: SUPERSEDED (historical text)"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(blocks)
