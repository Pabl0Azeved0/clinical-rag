PYTHON := .venv/bin/python
PIP    := .venv/bin/pip

.PHONY: venv install ingest search generate ui eval test format clean

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) scripts/ingest.py

search:
	@read -p "Query: " q; $(PYTHON) scripts/search.py "$$q"

generate:
	@read -p "Question: " q; $(PYTHON) scripts/generate.py "$$q"

ui:
	.venv/bin/streamlit run app/streamlit_app.py --server.fileWatcherType none

eval:
	.venv/bin/python evals/run_retrieval_eval.py

test:
	.venv/bin/pytest -q

format:
	.venv/bin/black src/ scripts/ tests/

clean:
	rm -rf .venv data/raw data/chroma __pycache__ .pytest_cache
