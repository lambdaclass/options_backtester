.PHONY: install env test test_notebook lint notebook help
.DEFAULT_GOAL := help

install: ## Create environment and install dependencies
	pipenv install && pipenv sync --dev

env: ## Run pipenv shell
	pipenv shell

notebook: ## Run Jupyter notebook
	pipenv run jupyter notebook

lint: ## Run linter and format checker (flake8 & yapf)
	pipenv run flake8 backtester
	pipenv run yapf --diff --recursive backtester/

test: ## Run tests
	pipenv run python -m pytest -v backtester

test_notebook: ## Run jupyter notebook test
	pipenv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=60 \
	backtester/examples/backtester_example.ipynb --stdout > /dev/null

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
