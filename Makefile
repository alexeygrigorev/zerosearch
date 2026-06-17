.PHONY: test setup shell coverage

test:
	uv run pytest

setup:
	uv sync --dev

shell:
	uv shell

coverage:
	uv run pytest --cov=zerosearch --cov-report=term-missing
