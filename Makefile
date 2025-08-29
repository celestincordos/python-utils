lint-types: lint build

lint:
	uv run ruff format . 
	uv run ruff check --fix . 
	uv run isort .

check-lint:
	uv run black . --check
	uv run ruff check . 
	uv run isort . --check

build: 
	uv run mypy --strict .
	uv export --no-hashes > requirements.txt

test: 
	uv run pytest --cov=. --cov-report=html

check: check-lint build test
all: lint build test