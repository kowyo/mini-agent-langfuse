.PHONY: install test lint clean

install:
	uv sync

test:
	uv run python -m pytest tests/ -v

lint:
	uv run ruff check src/ tests/

typing:
	uv run pyright src/

clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
