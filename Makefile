.PHONY: all test dist clean

PYTHON ?= python3

all: test dist

test:
	$(PYTHON) -m pytest -q

dist: clean
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf build dist *.egg-info
