PYTHON ?= python3
QUALITY_ARGS ?=

.PHONY: quality-env quality-warm hooks hooks-pre-push quality format test-cpu
quality-env:
	$(PYTHON) scripts/quality.py setup

quality-warm:
	$(PYTHON) scripts/quality.py warm

hooks:
	$(PYTHON) scripts/quality.py install-hooks

hooks-pre-push:
	$(PYTHON) scripts/quality.py install-hooks --pre-push

quality:
	$(PYTHON) scripts/quality.py check $(QUALITY_ARGS)

format:
	$(PYTHON) scripts/quality.py fix $(QUALITY_ARGS)

test-cpu:
	$(PYTHON) scripts/quality.py cpu
