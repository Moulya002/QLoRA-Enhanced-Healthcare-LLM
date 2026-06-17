# ============================================================================
# Developer workflow shortcuts. Run `make help` for the list.
# ============================================================================
.PHONY: help setup install install-dev data train evaluate infer api ui \
        docker-up docker-down test lint format typecheck clean

PYTHON ?= python
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt

install:  ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install dev/test dependencies
	$(PIP) install -r requirements-dev.txt

data:  ## Download + preprocess the dataset
	$(PY) -m data.preprocessing

train:  ## Run QLoRA fine-tuning (requires CUDA GPU)
	$(PY) -m src.train

train-smoke:  ## Quick training smoke test (tiny subset, no W&B)
	$(PY) -m src.train --max-train-samples 100 --epochs 1 --no-wandb

evaluate:  ## Evaluate base vs fine-tuned and write evaluation_report.md
	$(PY) -m evaluation.evaluate --num-samples 100

infer:  ## Interactive inference REPL
	$(PY) -m src.inference --interactive

api:  ## Run the FastAPI backend locally
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:  ## Run the Streamlit frontend locally
	$(PY) -m streamlit run app/streamlit_app.py

docker-up:  ## Build + start API and UI via docker compose
	docker compose up --build

docker-down:  ## Stop containers
	docker compose down

test:  ## Run unit tests
	$(PY) -m pytest

lint:  ## Lint with ruff
	$(VENV)/bin/ruff check .

format:  ## Auto-format with ruff
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

typecheck:  ## Static type check with mypy
	$(VENV)/bin/mypy src api evaluation experiments data

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ *.egg-info
