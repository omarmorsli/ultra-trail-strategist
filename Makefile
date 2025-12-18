.PHONY: install format lint test type-check clean check

install:
	pdm install

format:
	pdm run ruff format .
	pdm run ruff check --fix .

lint:
	pdm run ruff check .

test:
	pdm run pytest

type-check:
	pdm run mypy .

check: lint type-check test

clean:
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
