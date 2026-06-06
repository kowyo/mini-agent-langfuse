.PHONY: install test lint clean

install:
	pip install -e .

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

typing:
	pyright src/

clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
