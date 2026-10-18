NIX_CMD := XDG_CACHE_HOME=$(CURDIR)/.cache nix --extra-experimental-features 'nix-command flakes' develop --command
RUNCMD := $(NIX_CMD)
PYTHON := python

.PHONY: test test-heavy lint typecheck rust-build rust-test rust-bench install-dev help
.DEFAULT_GOAL := help

test: ## Run all tests
	$(RUNCMD) $(PYTHON) -m pytest -v tests

test-heavy: ## Run heavy at-scale correctness tests (needs data)
	$(RUNCMD) $(PYTHON) -m pytest -v -m heavy tests/heavy -o addopts=""

lint: ## Run ruff linter
	$(RUNCMD) $(PYTHON) -m ruff check options_portfolio_backtester tests scripts research

typecheck: ## Run mypy type checker
	$(RUNCMD) $(PYTHON) -m mypy options_portfolio_backtester --ignore-missing-imports

rust-build: ## Build Rust extension with maturin (release)
	$(RUNCMD) bash -c 'cd rust && maturin develop --manifest-path ob_python/Cargo.toml --release'

rust-test: ## Run Rust unit tests
	$(RUNCMD) bash -c 'cd rust && cargo test'

rust-bench: ## Run Rust benchmarks (criterion)
	$(RUNCMD) bash -c 'cd rust && cargo bench'

install-dev: ## Install local dev deps into active nix dev environment
	PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 $(PYTHON) -m pip install -e '.[dev,charts,rust]'

verify-articles: ## Reproduce all published-article tables and assert they still match
	$(RUNCMD) $(PYTHON) -m pytest -v tests/oracles/test_article_reproduction.py

verify-articles-smoke: ## Fast qualitative check (single budget, ~20s); suitable for pre-commit
	$(RUNCMD) $(PYTHON) -m pytest -v tests/oracles/test_article_reproduction.py -k smoke


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
