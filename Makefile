.PHONY: pilot all workload tune run-low run-high judge metrics audit test lint typecheck serve clean

CONFIG ?= config/bench.yaml
VENV ?= .venv/bin

pilot: workload tune
	$(VENV)/python -m lgb run --config baseline --frac low --limit 50
	$(VENV)/python -m lgb run --config cache --frac low --limit 50
	$(VENV)/python -m lgb run --config router_cascade --frac low --limit 50
	$(VENV)/python -m lgb run --config router_heuristic --frac low --limit 50
	$(VENV)/python -m lgb judge --config cache --frac low
	$(VENV)/python -m lgb judge --config router_cascade --frac low
	$(VENV)/python -m lgb judge --config router_heuristic --frac low
	$(VENV)/python -m lgb metrics --name pilot_low_50 --note "50-request pilot, low duplicate fraction"

workload:
	$(VENV)/python -m lgb workload

tune:
	$(VENV)/python -m lgb tune

run-low:
	$(VENV)/python -m lgb run --config baseline --frac low
	$(VENV)/python -m lgb run --config cache --frac low
	$(VENV)/python -m lgb run --config router_cascade --frac low
	$(VENV)/python -m lgb run --config router_heuristic --frac low

run-high:
	$(VENV)/python -m lgb run --config baseline --frac high
	$(VENV)/python -m lgb run --config cache --frac high
	$(VENV)/python -m lgb run --config router_cascade --frac high
	$(VENV)/python -m lgb run --config router_heuristic --frac high

judge:
	$(VENV)/python -m lgb judge --config cache --frac low
	$(VENV)/python -m lgb judge --config router_cascade --frac low
	$(VENV)/python -m lgb judge --config router_heuristic --frac low
	$(VENV)/python -m lgb judge --config cache --frac high
	$(VENV)/python -m lgb judge --config router_cascade --frac high
	$(VENV)/python -m lgb judge --config router_heuristic --frac high

metrics:
	$(VENV)/python -m lgb metrics --name full --note "full report-half runs, both duplicate fractions"

all: workload tune run-low run-high judge metrics audit

audit:
	$(VENV)/python scripts/audit_docs.py

test:
	$(VENV)/pytest -q

lint:
	$(VENV)/ruff check src tests scripts

typecheck:
	$(VENV)/mypy

serve:
	$(VENV)/python -m lgb serve

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage
