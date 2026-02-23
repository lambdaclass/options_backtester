/// Parallel grid sweep using Rayon with real run_backtest() per config.
///
/// Receives options+stocks data as DataFrames once, shares via Arc,
/// runs a full backtest per param override set in parallel.
/// No pickle overhead — data stays in shared memory.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use ob_core::backtest::{run_backtest, BacktestConfig, SchemaMapping};

use crate::arrow_bridge::py_to_polars;
use crate::py_backtest::{parse_config_from_dict, parse_schema};

/// Overrides parsed from each param dict (on GIL thread).
struct SweepOverrides {
    label: String,
    profit_pct: Option<Option<f64>>,   // None=use base, Some(None)=clear, Some(Some(v))=override
    loss_pct: Option<Option<f64>>,
    rebalance_dates: Option<Vec<String>>,
    leg_entry_filters: Option<Vec<Option<String>>>,
    leg_exit_filters: Option<Vec<Option<String>>>,
}

struct SweepResult {
    label: String,
    stats: ob_core::types::Stats,
    final_cash: f64,
    elapsed_ms: u128,
    error: Option<String>,
}

/// Merge base config with overrides, returning a new BacktestConfig.
fn merge_config(base: &BacktestConfig, overrides: &SweepOverrides) -> BacktestConfig {
    let mut cfg = base.clone();

    if let Some(ref pp) = overrides.profit_pct {
        cfg.profit_pct = *pp;
    }
    if let Some(ref lp) = overrides.loss_pct {
        cfg.loss_pct = *lp;
    }
    if let Some(ref dates) = overrides.rebalance_dates {
        cfg.rebalance_dates = dates.clone();
    }
    if let Some(ref filters) = overrides.leg_entry_filters {
        for (i, f) in filters.iter().enumerate() {
            if i < cfg.legs.len() {
                cfg.legs[i].entry_filter_query = f.clone();
            }
        }
    }
    if let Some(ref filters) = overrides.leg_exit_filters {
        for (i, f) in filters.iter().enumerate() {
            if i < cfg.legs.len() {
                cfg.legs[i].exit_filter_query = f.clone();
            }
        }
    }

    cfg
}

fn run_single_sweep(
    opts: &polars::prelude::DataFrame,
    stocks: &polars::prelude::DataFrame,
    base: &BacktestConfig,
    schema: &SchemaMapping,
    overrides: &SweepOverrides,
) -> SweepResult {
    let label = overrides.label.clone();
    let cfg = merge_config(base, overrides);
    let start = std::time::Instant::now();

    match run_backtest(&cfg, opts, stocks, schema) {
        Ok(result) => SweepResult {
            label,
            final_cash: result.final_cash,
            stats: result.stats,
            elapsed_ms: start.elapsed().as_millis(),
            error: None,
        },
        Err(e) => SweepResult {
            label,
            stats: Default::default(),
            final_cash: 0.0,
            elapsed_ms: start.elapsed().as_millis(),
            error: Some(format!("backtest error: {e}")),
        },
    }
}

/// Parse an optional f64 that may be absent, null, or a float.
/// None  → key missing (use base), Some(None) → explicit null (clear), Some(Some(v)) → override.
fn parse_opt_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Option<f64>>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(Some(None)),
        Some(v) => Ok(Some(Some(v.extract::<f64>()?))),
    }
}

/// Parse a single param override dict from Python.
fn parse_overrides(dict: &Bound<'_, PyDict>) -> PyResult<SweepOverrides> {
    let label = dict
        .get_item("label")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_default();

    // profit_pct/loss_pct: missing key → None (use base), None value → Some(None) (clear), float → Some(Some(v))
    let profit_pct = parse_opt_f64(dict, "profit_pct")?;
    let loss_pct = parse_opt_f64(dict, "loss_pct")?;

    let rebalance_dates: Option<Vec<String>> = dict
        .get_item("rebalance_dates")?
        .map(|v| v.extract::<Vec<String>>())
        .transpose()?;

    let leg_entry_filters: Option<Vec<Option<String>>> = dict
        .get_item("leg_entry_filters")?
        .map(|v| v.extract::<Vec<Option<String>>>())
        .transpose()?;

    let leg_exit_filters: Option<Vec<Option<String>>> = dict
        .get_item("leg_exit_filters")?
        .map(|v| v.extract::<Vec<Option<String>>>())
        .transpose()?;

    Ok(SweepOverrides {
        label,
        profit_pct,
        loss_pct,
        rebalance_dates,
        leg_entry_filters,
        leg_exit_filters,
    })
}

/// Run a parallel grid sweep over parameter combinations.
///
/// For each param dict, merges overrides into the base config and runs
/// a full backtest. All CPU-bound work runs on Rayon threads; only the
/// result collection touches the GIL.
///
/// Args:
///     options_data: Options DataFrame (shared across all workers)
///     stocks_data: Stocks DataFrame (shared across all workers)
///     base_config: Base backtest config dict
///     schema_mapping: Schema column name mappings dict
///     param_grid: List of override dicts, each with optional keys:
///         - "label": str (for identification)
///         - "profit_pct": Optional[float]
///         - "loss_pct": Optional[float]
///         - "rebalance_dates": Optional[list[str]]
///         - "leg_entry_filters": Optional[list[Optional[str]]]
///         - "leg_exit_filters": Optional[list[Optional[str]]]
///     n_workers: Number of Rayon threads (default: all cores)
///
/// Returns:
///     List of result dicts with stats for each parameter combination.
#[pyfunction]
#[pyo3(signature = (options_data, stocks_data, base_config, schema_mapping, param_grid, n_workers = None))]
pub fn parallel_sweep(
    py: Python<'_>,
    options_data: PyDataFrame,
    stocks_data: PyDataFrame,
    base_config: &Bound<'_, PyDict>,
    schema_mapping: &Bound<'_, PyDict>,
    param_grid: &Bound<'_, PyList>,
    n_workers: Option<usize>,
) -> PyResult<PyObject> {
    let opts = py_to_polars(options_data);
    let stocks = py_to_polars(stocks_data);

    // Parse base config and schema on GIL thread
    let base = parse_config_from_dict(base_config)?;
    let schema = parse_schema(schema_mapping)?;

    // Parse all override dicts on main thread (needs GIL)
    let overrides: Vec<SweepOverrides> = param_grid
        .iter()
        .map(|item| {
            let dict = item.downcast::<PyDict>().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("expected dict: {e}"))
            })?;
            parse_overrides(dict)
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Release GIL and run parallel computation with scoped Rayon pool
    let results: Vec<SweepResult> = py.allow_threads(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers.unwrap_or(0))
            .build()
            .expect("failed to build rayon thread pool");
        pool.install(|| {
            overrides
                .par_iter()
                .map(|ov| run_single_sweep(&opts, &stocks, &base, &schema, ov))
                .collect()
        })
    });

    // Convert results back to Python (needs GIL)
    let py_results = PyList::empty(py);
    for r in &results {
        let dict = PyDict::new(py);
        dict.set_item("label", &r.label)?;
        dict.set_item("total_return", r.stats.total_return)?;
        dict.set_item("annualized_return", r.stats.annualized_return)?;
        dict.set_item("sharpe_ratio", r.stats.sharpe_ratio)?;
        dict.set_item("sortino_ratio", r.stats.sortino_ratio)?;
        dict.set_item("calmar_ratio", r.stats.calmar_ratio)?;
        dict.set_item("max_drawdown", r.stats.max_drawdown)?;
        dict.set_item("max_drawdown_duration", r.stats.max_drawdown_duration)?;
        dict.set_item("profit_factor", r.stats.profit_factor)?;
        dict.set_item("win_rate", r.stats.win_rate)?;
        dict.set_item("total_trades", r.stats.total_trades)?;
        dict.set_item("final_cash", r.final_cash)?;
        dict.set_item("elapsed_ms", r.elapsed_ms)?;
        dict.set_item("error", &r.error)?;
        py_results.append(dict)?;
    }
    Ok(py_results.into())
}
