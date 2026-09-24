"""Test fixtures: every test runs against a throwaway SQLite DB + fresh index."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "corpus" / "seed"


@pytest.fixture(scope="session", autouse=True)
def _env(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["EMBEDDING_PROVIDER"] = "hashing"
    os.environ["RERANKER_PROVIDER"] = "lexical"
    os.environ["LLM_PROVIDER"] = "extractive"

    from core.config import get_settings
    from generation.llm import get_llm

    import core.config as config_mod
    import generation.answer as ans_mod
    import generation.llm as llm_mod

    config_mod.settings.llm_provider = "extractive"
    config_mod.settings.database_url = f"sqlite:///{db_path}"
    ans_mod.settings.llm_provider = "extractive"
    llm_mod.settings.llm_provider = "extractive"
    get_llm.cache_clear()
    yield


@pytest.fixture(scope="session")
def indexed(_env):
    """A session-scoped index over the whole seed corpus."""
    from core.db import init_db, reset_engine, session_scope
    from ingestion.pipeline import ingest_directory
    from retrieval.service import get_retrieval_service, reset_retrieval_service

    reset_engine()
    reset_retrieval_service()
    init_db()
    with session_scope() as session:
        results = ingest_directory(session, SEED)
    svc = get_retrieval_service()
    svc.mark_dirty()
    svc.ensure_index()
    return results


@pytest.fixture()
def client(indexed):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
