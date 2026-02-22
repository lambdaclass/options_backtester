NIX_CMD := nix --extra-experimental-features 'nix-command flakes' develop --command

.PHONY: test lint typecheck notebooks help
.DEFAULT_GOAL := help

test: ## Run tests
	$(NIX_CMD) python -m pytest -v backtester

lint: ## Run linter and format checker (flake8 & yapf)
	$(NIX_CMD) flake8 backtester
	$(NIX_CMD) yapf --diff --recursive backtester/

typecheck: ## Run mypy type checker
	$(NIX_CMD) python -m mypy backtester --ignore-missing-imports

notebooks: ## Execute all notebooks
	@for nb in notebooks/*.ipynb; do \
		echo "Running $$nb..."; \
		$(NIX_CMD) python -m jupyter nbconvert --to notebook --execute "$$nb" \
			--output "$$(basename $$nb)" --ExecutePreprocessor.timeout=600 || true; \
	done

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
