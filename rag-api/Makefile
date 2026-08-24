.PHONY: lint format evals

lint:
	ruff check . --fix

format:
	black .

evals:
	python scripts/run_evals.py --predictions evals/predictions.jsonl