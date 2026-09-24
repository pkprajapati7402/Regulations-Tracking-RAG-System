"""LLM client abstraction.

Providers: groq | gemini | openai | extractive (no-API fallback).

Every provider returns a ``LLMResponse`` carrying token usage, because the
project logs per-query token usage even on free tiers (see Project-Details.md
section 11).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache

import httpx

from core.config import settings
from core.logging_conf import get_logger

log = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM(ABC):
    provider: str

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse: ...

    @property
    def available(self) -> bool:
        return True


class ExtractiveLLM(BaseLLM):
    """Deterministic, key-free 'generation'.

    Rather than hallucinating fluent prose, it composes the answer out of the
    retrieved passages themselves. This keeps the whole system runnable (and
    the citation-faithfulness rate trivially high) without an API key, and it
    is what the test-suite exercises.
    """

    provider = "extractive"

    def complete(self, system: str, user: str) -> LLMResponse:  # pragma: no cover - unused
        return LLMResponse(text="", provider=self.provider, model="extractive")


class GroqLLM(BaseLLM):
    provider = "groq"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = httpx.post(
            self.endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        content = data["choices"][0]["message"].get("content") or ""
        return LLMResponse(
            text=content,
            provider=self.provider,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


class OpenAILLM(GroqLLM):
    provider = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"


class GeminiLLM(BaseLLM):
    provider = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model or "gemini-3.6-flash"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> LLMResponse:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": settings.llm_max_tokens,
            },
        }
        r = httpx.post(url, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        parts = data["candidates"][0]["content"]["parts"]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text="".join(p.get("text", "") for p in parts),
            provider=self.provider,
            model=self.model,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
        )


@lru_cache(maxsize=16)
def get_llm(provider: str | None = None, model: str | None = None) -> BaseLLM:
    provider = (provider or settings.llm_provider).lower()
    model = model or settings.llm_model
    client: BaseLLM
    if provider == "groq":
        client = GroqLLM(settings.groq_api_key, model)
    elif provider == "openai":
        client = OpenAILLM(settings.openai_api_key, model if "gpt" in model else "gpt-4o-mini")
    elif provider == "gemini":
        client = GeminiLLM(settings.gemini_api_key, model if "gemini" in model else "gemini-3.6-flash")
    else:
        return ExtractiveLLM()

    if not client.available:
        log.warning("LLM provider '%s' selected but no API key set; using extractive mode.", provider)
        return ExtractiveLLM()
    return client
