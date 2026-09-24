"""Citation faithfulness checking.

For every sentence of the generated answer that carries a citation, we check
whether the cited passage actually supports it. Two modes:

* lexical entailment (default): content-word overlap + numeric-token agreement
  between the claim and the cited passage. Numbers matter disproportionately in
  regulatory text ("within 30 days", "Rs 50,000"), so a claim that introduces a
  number absent from its cited passage is penalised hard.
* LLM-as-judge (when an LLM provider is configured and ``use_llm=True``),
  scoped narrowly to "does this passage support this claim".

The per-answer faithfulness rate is returned and surfaced in the API/UI, and
aggregated across the eval query set by ``evaluation/run_eval.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.config import settings
from generation.llm import ExtractiveLLM, get_llm
from generation.prompts import FAITHFULNESS_PROMPT

_CITE_RE = re.compile(r"(?:\[|\u3010)(\d{1,2})(?:\u2020[^\u3011]+)?(?:\]|\u3011)")
# Do not split before a trailing citation marker: "... detection. [1]" is one claim.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?!(?:\[|\u3010)\d)")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.%/,]*")
_NUM_RE = re.compile(r"\d[\d,\.]*")

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "fifteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "ninety", "hundred", "thousand", "lakh",
    "crore", "million", "first", "second", "third", "daily", "weekly",
    "monthly", "quarterly", "annually", "yearly",
}

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "by", "is",
    "are", "be", "as", "at", "with", "that", "this", "it", "shall", "may",
    "must", "should", "from", "which", "their", "its", "not", "per", "such",
    "any", "all", "under", "into", "these", "those", "have", "has", "been",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _numbers(text: str) -> set[str]:
    return {n.rstrip(".,").replace(",", "") for n in _NUM_RE.findall(text)}


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]
    supported: bool
    score: float
    method: str

    def as_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "cited": self.cited,
            "supported": self.supported,
            "score": round(self.score, 3),
            "method": self.method,
        }


@dataclass
class FaithfulnessReport:
    rate: float
    checked: int
    supported: int
    uncited_claims: int
    invalid_citations: list[int] = field(default_factory=list)
    sentences: list[SentenceCheck] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rate": round(self.rate, 3),
            "checked": self.checked,
            "supported": self.supported,
            "uncited_claims": self.uncited_claims,
            "invalid_citations": self.invalid_citations,
            "sentences": [s.as_dict() for s in self.sentences],
        }


def _salient(claim: str) -> set[str]:
    """Tokens a claim cannot invent: numerals, number words, and acronyms.

    Regulatory answers live or die on these ("within two to six hours",
    "report to FIU-IND"), so a claim carrying one that the cited passage does
    not contain is heavily penalised even if the surrounding prose overlaps.
    """
    tokens = set(_WORD_RE.findall(claim.lower()))
    salient = {t for t in tokens if t in _NUMBER_WORDS}
    salient |= {n.rstrip(".,").replace(",", "") for n in _NUM_RE.findall(claim)}
    salient |= {a.lower() for a in re.findall(r"\b[A-Z]{2,6}(?:-[A-Z]{2,6})?\b", claim)}
    return {s for s in salient if s}


def lexical_support(claim: str, passage: str) -> float:
    claim_words = _content_words(claim)
    if not claim_words:
        return 1.0
    passage_words = _content_words(passage)
    overlap = len(claim_words & passage_words) / len(claim_words)

    score = overlap
    claim_nums = _numbers(claim)
    if claim_nums:
        passage_nums = _numbers(passage)
        num_overlap = len(claim_nums & passage_nums) / len(claim_nums)
        score = 0.6 * overlap + 0.4 * num_overlap

    salient = _salient(claim)
    if salient:
        passage_lower = passage.lower()
        missing = [s for s in salient if s not in passage_lower]
        if missing:
            score *= max(0.15, 1.0 - 0.6 * len(missing) / len(salient))
    return score


def llm_support(claim: str, passage: str) -> bool | None:
    llm = get_llm()
    if isinstance(llm, ExtractiveLLM):
        return None
    try:
        resp = llm.complete(
            "You are a strict fact-checking judge. Reply with a single word.",
            FAITHFULNESS_PROMPT.format(claim=claim, passage=passage[:4000]),
        )
        return "UNSUPPORTED" not in resp.text.upper()
    except Exception:  # pragma: no cover - network failure -> fall back
        return None


def check_answer(answer: str, sources: list, use_llm: bool = False) -> FaithfulnessReport:
    """``sources`` is the ordered list of RetrievedChunk objects given to the LLM."""
    # Normalize thin / non-breaking spaces
    sentences: list[str] = []
    normalized = (answer or "").replace("\u202f", " ").replace("\u00a0", " ")
    for line in normalized.split("\n"):
        sentences.extend(s.strip(" -*\t") for s in _SENT_SPLIT.split(line) if s.strip())
    checks: list[SentenceCheck] = []
    invalid: list[int] = []
    uncited = 0

    for sentence in sentences:
        cites = [int(m) for m in _CITE_RE.findall(sentence)]
        bare = _CITE_RE.sub("", sentence).strip()
        if len(_content_words(bare)) < 3:
            continue  # boilerplate / disclaimer / heading
        if not cites:
            if (
                bare.lower().startswith(("this is informational", "the indexed circulars do not"))
                or bare.endswith(":")
                or (bare.startswith("**") and bare.endswith("**"))
            ):
                continue
            uncited += 1
            checks.append(SentenceCheck(sentence, [], False, 0.0, "uncited"))
            continue

        best = 0.0
        method = "lexical"
        supported = False
        for idx in cites:
            if idx < 1 or idx > len(sources):
                invalid.append(idx)
                continue
            passage = sources[idx - 1].text
            if use_llm:
                verdict = llm_support(bare, passage)
                if verdict is not None:
                    method = "llm-judge"
                    supported = supported or verdict
                    best = max(best, 1.0 if verdict else 0.0)
                    continue
            score = lexical_support(bare, passage)
            best = max(best, score)
            supported = supported or score >= settings.faithfulness_threshold
        checks.append(SentenceCheck(sentence, cites, supported, best, method))

    checked = len(checks)
    supported_count = sum(1 for c in checks if c.supported)
    rate = supported_count / checked if checked else 1.0
    return FaithfulnessReport(
        rate=rate,
        checked=checked,
        supported=supported_count,
        uncited_claims=uncited,
        invalid_citations=sorted(set(invalid)),
        sentences=checks,
    )
