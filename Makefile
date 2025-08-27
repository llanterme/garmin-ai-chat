.PHONY: help install install-dev test test-cov lint format check run clean migrate upgrade downgrade docker-up docker-down

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	uv sync --no-dev

install-dev: ## Install all dependencies including development
	uv sync

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html --cov-report=term

lint: ## Run linting checks
	uv run ruff check src tests
	uv run mypy src

format: ## Format code
	uv run black src tests
	uv run ruff format src tests

check: ## Run all checks (lint, format, test)
	make format
	make lint
	make test

run: ## Run the development server
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

migrate: ## Generate new migration
	uv run alembic revision --autogenerate -m "$(MESSAGE)"

upgrade: ## Run database migrations
	uv run alembic upgrade head

downgrade: ## Rollback database migration
	uv run alembic downgrade -1

docker-up: ## Start MySQL container
	docker run --name garmin-mysql -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=garmin_ai_chat -p 3306:3306 -d mysql:8.0

docker-down: ## Stop MySQL container
	docker stop garmin-mysql && docker rm garmin-mysql

init: ## Initialize project (install deps, setup db, run migrations)
	make install-dev
	make docker-up
	sleep 10
	make upgrade