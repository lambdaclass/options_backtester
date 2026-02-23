NIX_CMD := nix --extra-experimental-features 'nix-command flakes' develop --command

.PHONY: test test-old test-new lint typecheck notebooks help
.DEFAULT_GOAL := help

test: ## Run all tests (old + new)
	$(NIX_CMD) python -m pytest -v backtester/test tests

test-old: ## Run legacy tests only
	$(NIX_CMD) python -m pytest -v backtester/test

test-new: ## Run new framework tests only
	$(NIX_CMD) python -m pytest -v tests

lint: ## Run ruff linter
	$(NIX_CMD) python -m ruff check backtester options_backtester

typecheck: ## Run mypy type checker
	$(NIX_CMD) python -m mypy backtester options_backtester --ignore-missing-imports

notebooks: ## Execute all notebooks
	@for nb in notebooks/*.ipynb; do \
		echo "Running $$nb..."; \
		$(NIX_CMD) python -m jupyter nbconvert --to notebook --execute "$$nb" \
			--output "$$(basename $$nb)" --ExecutePreprocessor.timeout=600 || true; \
	done

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
