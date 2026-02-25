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
use ob_core::cost_model::CostModel;
use ob_core::fill_model::FillModel;
use ob_core::risk::RiskConstraint;
use ob_core::signal_selector::SignalSelector;

use crate::arrow_bridge::py_to_polars;
use crate::py_backtest::{
    parse_config_from_dict, parse_cost_model, parse_fill_model,
    parse_risk_constraint, parse_schema, parse_signal_selector,
};

/// Overrides parsed from each param dict (on GIL thread).
struct SweepOverrides {
    label: String,
    profit_pct: Option<Option<f64>>,   // None=use base, Some(None)=clear, Some(Some(v))=override
    loss_pct: Option<Option<f64>>,
    rebalance_dates: Option<Vec<i64>>,
    leg_entry_filters: Option<Vec<Option<String>>>,
    leg_exit_filters: Option<Vec<Option<String>>>,
    cost_model: Option<CostModel>,
    fill_model: Option<FillModel>,
    signal_selector: Option<SignalSelector>,
    risk_constraints: Option<Vec<RiskConstraint>>,
    sma_days: Option<Option<usize>>,
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
    if let Some(ref cm) = overrides.cost_model {
        cfg.cost_model = cm.clone();
    }
    if let Some(ref fm) = overrides.fill_model {
        cfg.fill_model = fm.clone();
    }
    if let Some(ref ss) = overrides.signal_selector {
        cfg.signal_selector = ss.clone();
    }
    if let Some(ref rc) = overrides.risk_constraints {
        cfg.risk_constraints = rc.clone();
    }
    if let Some(ref sma) = overrides.sma_days {
        cfg.sma_days = *sma;
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
/// None  -> key missing (use base), Some(None) -> explicit null (clear), Some(Some(v)) -> override.
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

    let profit_pct = parse_opt_f64(dict, "profit_pct")?;
    let loss_pct = parse_opt_f64(dict, "loss_pct")?;

    let rebalance_dates: Option<Vec<i64>> = dict
        .get_item("rebalance_dates")?
        .map(|v| v.extract::<Vec<i64>>())
        .transpose()?;

    let leg_entry_filters: Option<Vec<Option<String>>> = dict
        .get_item("leg_entry_filters")?
        .map(|v| v.extract::<Vec<Option<String>>>())
        .transpose()?;

    let leg_exit_filters: Option<Vec<Option<String>>> = dict
        .get_item("leg_exit_filters")?
        .map(|v| v.extract::<Vec<Option<String>>>())
        .transpose()?;

    let cost_model = match dict.get_item("cost_model")? {
        Some(v) if !v.is_none() => {
            let d = v.downcast::<PyDict>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;
            Some(parse_cost_model(d)?)
        }
        _ => None,
    };

    let fill_model = match dict.get_item("fill_model")? {
        Some(v) if !v.is_none() => {
            let d = v.downcast::<PyDict>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;
            Some(parse_fill_model(d)?)
        }
        _ => None,
    };

    let signal_selector = match dict.get_item("signal_selector")? {
        Some(v) if !v.is_none() => {
            let d = v.downcast::<PyDict>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;
            Some(parse_signal_selector(d)?)
        }
        _ => None,
    };

    let risk_constraints: Option<Vec<RiskConstraint>> = match dict.get_item("risk_constraints")? {
        Some(v) if !v.is_none() => {
            let list = v.downcast::<PyList>()
                .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;
            Some(list.iter()
                .map(|item| {
                    let d = item.downcast::<PyDict>()
                        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;
                    parse_risk_constraint(d)
                })
                .collect::<PyResult<Vec<_>>>()?)
        }
        _ => None,
    };

    let sma_days: Option<Option<usize>> = match dict.get_item("sma_days")? {
        None => None,
        Some(v) if v.is_none() => Some(None),
        Some(v) => Some(Some(v.extract::<usize>()?)),
    };

    Ok(SweepOverrides {
        label,
        profit_pct,
        loss_pct,
        rebalance_dates,
        leg_entry_filters,
        leg_exit_filters,
        cost_model,
        fill_model,
        signal_selector,
        risk_constraints,
        sma_days,
    })
}

/// Run a parallel grid sweep over parameter combinations.
///
/// For each param dict, merges overrides into the base config and runs
/// a full backtest. All CPU-bound work runs on Rayon threads; only the
/// result collection touches the GIL.
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
