# ============================================================
# AI Fairness Platform — Developer CLI
# ============================================================
.PHONY: help install dev test lint format build clean train eval report

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	cd backend && pip install -r requirements.txt -r requirements-dev.txt
	cd frontend && npm install --legacy-peer-deps

dev: ## Start full stack via Docker
	docker compose up -d --build

dev-backend: ## Run backend with hot-reload
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend: ## Run frontend dev server
	cd frontend && npm start

test: ## Run all backend tests with coverage
	cd backend && pytest -v --cov=app --cov-report=term-missing --cov-report=html

lint: ## Lint all Python code
	cd backend && ruff check app tests && black --check app tests

format: ## Auto-format Python code
	cd backend && black app tests && isort app tests

build: ## Build all Docker images
	docker compose build

clean: ## Stop containers and remove volumes
	docker compose down -v --remove-orphans

data-download: ## Download all benchmark datasets
	python datasets/scripts/download_datasets.py --all

data-preprocess: ## Preprocess all datasets
	python scripts/preprocess_datasets.py --all

run-analysis: ## Run a quick fairness analysis (Adult + LR)
	curl -s -X POST http://localhost:8000/api/v1/analysis/run \
	  -H "Content-Type: application/json" \
	  -d '{"dataset_type":"adult","model_type":"logistic_regression"}' | python3 -m json.tool

report: ## Generate a fairness audit report for last session
	@echo "Use POST /api/v1/reports/generate with your session_id"

notebooks: ## Launch Jupyter with notebooks
	jupyter lab --notebook-dir=notebooks --port=8888
