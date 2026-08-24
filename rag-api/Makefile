.PHONY: lint format evals reports

lint:
	ruff check . --fix

format:
	black .

evals:
	python scripts/run_evals.py --predictions evals/predictions.jsonl

reports:
	python scripts/generate_reports.py