/// Parallel grid sweep using Rayon.
///
/// Receives parameter grid as list of dicts, shares data via Arc,
/// runs each backtest in parallel. No pickle overhead.

use pyo3::prelude::*;

/// Parallel grid sweep placeholder.
///
/// In full implementation, this would:
/// 1. Receive options+stocks data as Arrow IPC bytes once
/// 2. Deserialize once, share via Arc across Rayon threads
/// 3. Each worker runs a full Rust-native backtest
/// 4. Return results as list of dicts
#[pyfunction]
#[pyo3(signature = (param_grid, n_workers = None))]
pub fn parallel_sweep(
    param_grid: Vec<PyObject>,
    n_workers: Option<usize>,
) -> PyResult<Vec<PyObject>> {
    let _pool_size = n_workers.unwrap_or_else(|| rayon::current_num_threads());

    // Placeholder: return empty results
    // Full implementation would use rayon::par_iter to run backtests
    Python::with_gil(|py| {
        let results: Vec<PyObject> = param_grid
            .iter()
            .map(|_params| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("status", "not_implemented").unwrap();
                dict.into()
            })
            .collect();
        Ok(results)
    })
}
