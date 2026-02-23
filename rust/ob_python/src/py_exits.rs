/// PyO3 bindings for exit mask computation.

use pyo3::prelude::*;
use polars::prelude::{NamedFrom, Series};

use ob_core::exits;

/// Compute threshold exit mask from entry and current costs.
#[pyfunction]
#[pyo3(signature = (entry_costs, current_costs, profit_pct = None, loss_pct = None))]
pub fn compute_exit_mask(
    entry_costs: Vec<f64>,
    current_costs: Vec<f64>,
    profit_pct: Option<f64>,
    loss_pct: Option<f64>,
) -> PyResult<Vec<bool>> {
    let entry_series = Series::new("entry".into(), &entry_costs);
    let current_series = Series::new("current".into(), &current_costs);

    let mask = exits::threshold_exit_mask(&entry_series, &current_series, profit_pct, loss_pct)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(mask.into_no_null_iter().collect())
}
