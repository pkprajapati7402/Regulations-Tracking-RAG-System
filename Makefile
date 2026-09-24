.PHONY: help install install-ml seed serve test eval eval-fast case-study status fetch docker clean

PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install: ## create venv and install runtime + dev deps
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip wheel
	$(BIN)/pip install -r requirements.txt
	cp -n .env.example .env || true

install-ml: ## add neural embeddings + cross-encoder reranker (large download)
	$(BIN)/pip install -r requirements-ml.txt

seed: ## index the offline seed corpus
	$(BIN)/python -m ingestion.reindex --reset --seed

fetch: ## download real RBI circulars into corpus/raw (needs network)
	$(BIN)/python -m ingestion.fetch_rbi --category master-circulars --limit 40 --out corpus/raw
	$(BIN)/python -m ingestion.reindex --path corpus/raw

serve: ## run the API + chat UI on :8000
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test: ## run the test-suite
	$(BIN)/python -m pytest

eval: ## run the full evaluation suite (retrieval + chunking + faithfulness)
	$(BIN)/python -m evaluation.run_eval --k 5

eval-fast: ## retrieval comparison only
	$(BIN)/python -m evaluation.run_eval --experiment retrieval --k 5

case-study: ## reproduce the amendment / stale-data case study
	$(BIN)/python scripts/case_study_amendment.py

status: ## show what is indexed
	$(BIN)/python -m ingestion.reindex --status

docker: ## build and run with docker compose
	docker compose up --build

clean:
	rm -rf data/*.db data/uploads .pytest_cache **/__pycache__
