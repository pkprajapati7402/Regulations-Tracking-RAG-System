"""Answer generation with forced citations + conflict/amendment awareness."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from core.config import settings
from core.logging_conf import get_logger
from generation.faithfulness import FaithfulnessReport, check_answer
from generation.llm import ExtractiveLLM, LLMResponse, get_llm
from generation.prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT, format_sources
from retrieval.types import RetrievedChunk

log = get_logger(__name__)

NO_CONTEXT_ANSWER = "The indexed circulars do not contain enough information to answer this."
DISCLAIMER = "This is informational only, not legal or financial advice."

_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "by", "is",
    "are", "be", "as", "at", "with", "that", "this", "it", "what", "which",
    "how", "when", "does", "do", "under", "shall", "must", "who", "whom",
}


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[dict]
    faithfulness: FaithfulnessReport
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    notices: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "faithfulness": self.faithfulness.as_dict(),
            "usage": {
                "provider": self.provider,
                "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "latency_ms": round(self.latency_ms, 2),
            },
            "notices": self.notices,
        }


def _amendment_notices(chunks: list[RetrievedChunk]) -> list[str]:
    notices: list[str] = []
    superseded = sorted({c.doc_number for c in chunks if c.status != "active"})
    if superseded:
        notices.append(
            "Superseded text was included at your request (audit mode): "
            + ", ".join(superseded)
        )
    dates = [c.published_date for c in chunks if c.published_date]
    if len(set(dates)) > 1:
        notices.append(
            f"Sources span multiple publication dates ({min(dates)} to {max(dates)}); "
            "the later document prevails where they differ."
        )
    return notices


def _keywords(question: str) -> set[str]:
    return {w for w in _WORD_RE.findall(question.lower()) if w not in _STOP and len(w) > 2}


def extractive_answer(question: str, chunks: list[RetrievedChunk], max_points: int = 5) -> str:
    """Build a cited answer straight from the passages (no API key required)."""
    keywords = _keywords(question)
    scored: list[tuple[float, int, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        body = re.sub(r"^\[Section:[^\]]*\]\s*", "", chunk.text)
        for sentence in _SENT_SPLIT.split(body.replace("\n", " ")):
            sentence = sentence.strip(" -•\t")
            if len(sentence) < 40 or len(sentence) > 600:
                continue
            words = set(_WORD_RE.findall(sentence.lower()))
            hit = len(keywords & words)
            if not hit:
                continue
            density = hit / (1 + len(words) / 60)
            # Prefer sentences from higher-ranked chunks.
            scored.append((density + (len(chunks) - idx) * 0.01, idx, sentence))

    if not scored:
        return NO_CONTEXT_ANSWER + f"\n\n{DISCLAIMER}"

    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    lines: list[str] = []
    for _, idx, sentence in scored:
        key = sentence[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        clean = sentence.rstrip(".") + f". [{idx}]"
        lines.append(f"- {clean}")
        if len(lines) >= max_points:
            break

    head = chunks[0]
    header = f"Based on {head.doc_number} — {head.doc_title.rstrip('.')}, the indexed circulars state: [1]"
    return "\n".join([header, "", *lines, "", DISCLAIMER])


def retrieval_confidence(question: str, chunks: list[RetrievedChunk], top_n: int = 3) -> float:
    """Fraction of the question's content words covered by the top passages.

    This is the refusal gate: retrieval always returns *something*, so without
    a confidence floor an out-of-domain question would be answered from
    loosely-related text. Below the floor the system refuses instead.
    """
    keywords = _keywords(question)
    if not keywords:
        return 1.0
    covered: set[str] = set()
    for chunk in chunks[:top_n]:
        words = set(_WORD_RE.findall(chunk.text.lower()))
        covered |= keywords & words
    return len(covered) / len(keywords)


MIN_CONFIDENCE = 0.4


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    provider: str | None = None,
    model: str | None = None,
    use_llm_judge: bool = False,
) -> GeneratedAnswer:
    started = time.perf_counter()
    notices = _amendment_notices(chunks)

    if not chunks:
        answer = f"{NO_CONTEXT_ANSWER}\n\n{DISCLAIMER}"
        return GeneratedAnswer(
            answer=answer,
            citations=[],
            faithfulness=check_answer(answer, []),
            provider="none",
            model="none",
            latency_ms=(time.perf_counter() - started) * 1000,
            notices=notices + ["No matching passages were retrieved. Try uploading the relevant circular."],
        )

    confidence = retrieval_confidence(question, chunks)
    if confidence < MIN_CONFIDENCE:
        answer = f"{NO_CONTEXT_ANSWER}\n\n{DISCLAIMER}"
        return GeneratedAnswer(
            answer=answer,
            citations=[],
            faithfulness=check_answer(answer, []),
            provider="none",
            model="none",
            latency_ms=(time.perf_counter() - started) * 1000,
            notices=notices
            + [
                f"Retrieval confidence {confidence:.0%} is below the {MIN_CONFIDENCE:.0%} floor - "
                "the indexed circulars do not appear to cover this topic."
            ],
        )

    llm = get_llm(provider=provider, model=model)
    if isinstance(llm, ExtractiveLLM):
        text = extractive_answer(question, chunks)
        resp = LLMResponse(text=text, provider="extractive", model="extractive-v1")
    else:
        prompt = ANSWER_TEMPLATE.format(question=question, sources=format_sources(chunks))
        try:
            resp = llm.complete(SYSTEM_PROMPT, prompt)
        except Exception as exc:
            log.warning("LLM call failed (%s); falling back to extractive answer.", exc)
            resp = LLMResponse(
                text=extractive_answer(question, chunks),
                provider="extractive",
                model="extractive-v1",
            )
            notices.append("LLM unavailable - answer composed extractively from the sources.")
        if DISCLAIMER.lower() not in resp.text.lower():
            resp.text = resp.text.rstrip() + f"\n\n{DISCLAIMER}"

    report = check_answer(resp.text, chunks, use_llm=use_llm_judge)
    if report.rate < 0.5 and report.checked:
        notices.append(
            f"Low citation-faithfulness score ({report.rate:.0%}); verify against the primary source."
        )

    citations = []
    for i, chunk in enumerate(chunks, start=1):
        data = chunk.as_dict()
        data["n"] = i
        data["snippet"] = re.sub(r"\s+", " ", chunk.text)[:400]
        citations.append(data)

    return GeneratedAnswer(
        answer=resp.text.strip(),
        citations=citations,
        faithfulness=report,
        provider=resp.provider,
        model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=(time.perf_counter() - started) * 1000,
        notices=notices,
    )
