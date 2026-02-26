NIX_CMD := XDG_CACHE_HOME=$(CURDIR)/.cache nix --extra-experimental-features 'nix-command flakes' develop --command
RUNCMD := $(NIX_CMD)
PYTHON := python

.PHONY: test test-bench lint typecheck notebooks rust-build rust-test rust-bench bench install-dev compare-bt benchmark-matrix walk-forward-report parity-gate bench-rust-vs-python help
.DEFAULT_GOAL := help

test: ## Run all tests
	$(RUNCMD) $(PYTHON) -m pytest -v tests

test-bench: ## Run benchmark/property tests
	$(RUNCMD) $(PYTHON) -m pytest -v -m bench tests/bench

lint: ## Run ruff linter
	$(RUNCMD) $(PYTHON) -m ruff check options_portfolio_backtester

typecheck: ## Run mypy type checker
	$(RUNCMD) $(PYTHON) -m mypy options_portfolio_backtester --ignore-missing-imports

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
	$(RUNCMD) $(PYTHON) -m pytest tests/bench/ -v -m bench --benchmark-only 2>/dev/null || \
		echo "Install pytest-benchmark for Python benchmarks"

install-dev: ## Install local dev deps into active nix dev environment
	$(PYTHON) -m pip install -e '.[dev,charts,notebooks,rust]'

compare-bt: ## Compare stock-only monthly rebalance vs bt library
	$(RUNCMD) $(PYTHON) scripts/compare_with_bt.py

benchmark-matrix: ## Run standardized runtime/accuracy matrix vs bt
	$(RUNCMD) $(PYTHON) scripts/benchmark_matrix.py

walk-forward-report: ## Run walk-forward/OOS harness and save report
	$(RUNCMD) $(PYTHON) scripts/walk_forward_report.py

bench-rust-vs-python: ## Benchmark Rust vs Python vs bt (options + stock-only)
	$(RUNCMD) $(PYTHON) scripts/benchmark_rust_vs_python.py --stock-only

parity-gate: ## Run bt overlap parity CI gate (bench marker)
	$(RUNCMD) $(PYTHON) -m pytest -v tests/compat/test_bt_overlap_gate.py -m bench

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
