PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

.PHONY: all download test study clean

all: test study

download:
	PYTHONPATH=src $(PYTHON) -m agent_loops.download

test:
	$(PYTEST) -q

study:
	PYTHONPATH=src $(PYTHON) run_all.py

clean:
	rm -rf results/* assets/*
