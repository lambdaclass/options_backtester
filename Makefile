.PHONY: install env test test_notebook
.DEFAULT_GOAL := help

install: ## Create environment and install dependencies
	pipenv --three && pipenv sync --dev

env: ## Run pipenv shell
	pipenv shell

test: ## Run tests
	pipenv run python -m pytest -v backtester

test_notebook: ## Run jupyter notebook test
	pipenv run jupyter nbconvert nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 \
	backtester/examples/backtester_example.ipynb --stdout > /dev/null

lint: ## Run linter and format checker (flake8 & yapf)
	pipenv run flake8 backtester
	pipenv run yapf --diff --recursive backtester/

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
