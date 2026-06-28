PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install ingest search test format clean

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) scripts/ingest.py

search:
	@read -p "Query: " q; $(PYTHON) scripts/search.py "$$q"

test:
	.venv/bin/pytest -q

format:
	.venv/bin/black src/ scripts/ tests/

clean:
	rm -rf .venv data/raw data/chroma __pycache__ .pytest_cache
