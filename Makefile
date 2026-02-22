.PHONY: test lint help
.DEFAULT_GOAL := help

test: ## Run tests
	python -m pytest -v backtester

lint: ## Run linter and format checker (flake8 & yapf)
	flake8 backtester
	yapf --diff --recursive backtester/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
