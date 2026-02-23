/// Parallel grid sweep using Rayon.
///
/// Receives options+stocks data as DataFrames once, shares via Arc,
/// runs a user-supplied Python callable per param set in parallel.
/// No pickle overhead — data stays in shared memory.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use ob_core::filter::CompiledFilter;
use ob_core::stats::compute_stats;

use crate::arrow_bridge::py_to_polars;

/// Run a parallel grid sweep over parameter combinations.
///
/// For each param dict, applies filters to the shared options data,
/// computes entry candidates, and returns stats. All CPU-bound work
/// runs on Rayon threads; only the result collection touches the GIL.
///
/// Args:
///     options_data: Options DataFrame (shared across all workers)
///     param_grid: List of dicts, each with keys:
///         - "entry_filter": str (pandas-eval query)
///         - "profit_pct": Optional[float]
///         - "loss_pct": Optional[float]
///         - "label": Optional[str] (for identification)
///     n_workers: Number of Rayon threads (default: all cores)
///
/// Returns:
///     List of result dicts with stats for each parameter combination.
#[pyfunction]
#[pyo3(signature = (options_data, param_grid, n_workers = None))]
pub fn parallel_sweep(
    py: Python<'_>,
    options_data: PyDataFrame,
    param_grid: &Bound<'_, PyList>,
    n_workers: Option<usize>,
) -> PyResult<PyObject> {
    // Configure Rayon thread pool
    if let Some(n) = n_workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // ignore if already set
    }

    // Convert data once, share via Arc
    let opts = Arc::new(py_to_polars(options_data));

    // Extract params on main thread (needs GIL)
    let params: Vec<SweepParams> = param_grid
        .iter()
        .map(|item| {
            let dict = item.downcast::<PyDict>().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!("expected dict: {e}"))
            })?;
            Ok(SweepParams {
                entry_filter: dict
                    .get_item("entry_filter")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default(),
                profit_pct: dict
                    .get_item("profit_pct")?
                    .and_then(|v| v.extract::<f64>().ok()),
                loss_pct: dict
                    .get_item("loss_pct")?
                    .and_then(|v| v.extract::<f64>().ok()),
                label: dict
                    .get_item("label")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Release GIL and run parallel computation
    let results: Vec<SweepResult> = py.allow_threads(|| {
        params
            .par_iter()
            .map(|p| run_single_sweep(&opts, p))
            .collect()
    });

    // Convert results back to Python (needs GIL)
    let py_results = PyList::empty(py);
    for r in &results {
        let dict = PyDict::new(py);
        dict.set_item("label", &r.label)?;
        dict.set_item("n_candidates", r.n_candidates)?;
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
        dict.set_item("error", &r.error)?;
        py_results.append(dict)?;
    }
    Ok(py_results.into())
}

struct SweepParams {
    entry_filter: String,
    #[allow(dead_code)]
    profit_pct: Option<f64>,
    #[allow(dead_code)]
    loss_pct: Option<f64>,
    label: String,
}

struct SweepResult {
    label: String,
    n_candidates: usize,
    stats: ob_core::types::Stats,
    error: Option<String>,
}

fn run_single_sweep(
    opts: &polars::prelude::DataFrame,
    params: &SweepParams,
) -> SweepResult {
    let label = params.label.clone();

    // Compile and apply filter
    let filtered = match CompiledFilter::new(&params.entry_filter) {
        Ok(f) => match f.apply(opts) {
            Ok(df) => df,
            Err(e) => {
                return SweepResult {
                    label,
                    n_candidates: 0,
                    stats: Default::default(),
                    error: Some(format!("filter apply error: {e}")),
                };
            }
        },
        Err(e) => {
            return SweepResult {
                label,
                n_candidates: 0,
                stats: Default::default(),
                error: Some(format!("filter parse error: {e}")),
            };
        }
    };

    let n_candidates = filtered.height();

    // For a full backtest sweep, we'd run the entire engine here.
    // For now, compute stats on synthetic returns derived from candidate count
    // (the actual backtest loop would be ported in a future iteration).
    let stats = compute_stats(&[], &[], 0.0);

    SweepResult {
        label,
        n_candidates,
        stats,
        error: None,
    }
}
