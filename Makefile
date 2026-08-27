PYTHON := .venv/bin/python
JUPYTER := .venv/bin/jupyter

.PHONY: install data audit ml test notebook app

install:
	python3.11 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install -e .

data:
	$(PYTHON) -m spamguard.data

audit:
	$(PYTHON) -m spamguard.audit

ml:
	$(PYTHON) -m spamguard.ml

test:
	$(PYTHON) -m pytest

notebook:
	$(JUPYTER) lab

app:
	.venv/bin/streamlit run app/streamlit_app.py
