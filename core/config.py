"""Central, environment-driven configuration.

Everything in the system reads its knobs from here, so a deployment can be
re-tuned (different embedder, reranker, LLM, chunker) purely through env vars.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- app ----
    app_name: str = "Regulation-Tracking RAG System"
    environment: str = "local"
    log_level: str = "INFO"

    # ---- storage ----
    database_url: str = f"sqlite:///{ROOT / 'data' / 'regrag.db'}"
    corpus_dir: Path = ROOT / "corpus"
    upload_dir: Path = ROOT / "data" / "uploads"

    # ---- embeddings ----
    embedding_provider: str = "hashing"  # hashing | sentence-transformers
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 1024  # only used by the hashing provider

    # ---- reranking ----
    reranker_provider: str = "lexical"  # lexical | cross-encoder
    reranker_model: str = "BAAI/bge-reranker-base"

    # ---- generation ----
    llm_provider: str = "extractive"  # extractive | groq | gemini | openai
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 900

    # ---- chunking ----
    chunker: str = "structure_aware"  # fixed | recursive | structure_aware
    chunk_tokens: int = 350
    chunk_overlap: int = 60

    # ---- retrieval ----
    retrieval_mode: str = "hybrid_rerank"  # bm25 | dense | hybrid | hybrid_rerank
    top_k: int = 8
    candidate_k: int = 40
    rrf_k: int = 60
    min_rerank_score: float = 0.0

    # ---- generation guard rails ----
    faithfulness_threshold: float = 0.35

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite"):
            db_path = self.database_url.split("///")[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
