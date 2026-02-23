NIX_CMD := XDG_CACHE_HOME=$(CURDIR)/.cache nix --extra-experimental-features 'nix-command flakes' develop --command
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
NIX_AVAILABLE := $(shell command -v nix >/dev/null 2>&1 && XDG_CACHE_HOME=$(CURDIR)/.cache nix show-config >/dev/null 2>&1 && echo 1 || true)

ifeq ($(NIX_AVAILABLE),1)
RUNCMD := $(NIX_CMD)
else
RUNCMD :=
endif

.PHONY: test test-old test-new test-bench lint typecheck notebooks rust-build rust-test rust-bench bench install-dev help
.DEFAULT_GOAL := help

test: ## Run all tests (old + new)
	$(RUNCMD) $(PYTHON) -m pytest -v backtester/test tests --ignore=tests/bench

test-old: ## Run legacy tests only
	$(RUNCMD) $(PYTHON) -m pytest -v backtester/test

test-new: ## Run new framework tests only
	$(RUNCMD) $(PYTHON) -m pytest -v tests --ignore=tests/bench

test-bench: ## Run benchmark/property tests
	$(RUNCMD) $(PYTHON) -m pytest -v tests/bench

lint: ## Run ruff linter
	$(RUNCMD) $(PYTHON) -m ruff check backtester options_backtester

typecheck: ## Run mypy type checker
	$(RUNCMD) $(PYTHON) -m mypy backtester options_backtester --ignore-missing-imports

notebooks: ## Execute all notebooks
	@for nb in notebooks/*.ipynb; do \
		echo "Running $$nb..."; \
		$(RUNCMD) $(PYTHON) -m jupyter nbconvert --to notebook --execute "$$nb" \
			--output "$$(basename $$nb)" --ExecutePreprocessor.timeout=600 || true; \
	done

rust-build: ## Build Rust extension with maturin (release)
	$(RUNCMD) maturin develop --manifest-path rust/ob_python/Cargo.toml --release

rust-test: ## Run Rust unit tests
	$(RUNCMD) bash -c 'cd rust && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test'

rust-bench: ## Run Rust benchmarks (criterion)
	$(RUNCMD) bash -c 'cd rust && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo bench'

bench: rust-build ## Run Python benchmarks (requires Rust build)
	$(RUNCMD) $(PYTHON) -m pytest tests/bench/ -v --benchmark-only 2>/dev/null || \
		echo "Install pytest-benchmark for Python benchmarks"

install-dev: ## Install local dev deps into current Python env
	$(PYTHON) -m pip install -e '.[dev,charts,notebooks,rust]'

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
