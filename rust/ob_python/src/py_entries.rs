//! PyO3 bindings for entry signal computation.

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use ob_core::entries;
use ob_core::filter;

use crate::arrow_bridge::{polars_to_py, py_to_polars};

#[pyfunction]
#[pyo3(signature = (
    options_data,
    inventory_contracts,
    entry_filter_query,
    contract_col,
    cost_field,
    entry_sort_col,
    entry_sort_asc,
    shares_per_contract,
    is_sell,
))]
pub fn compute_entries(
    options_data: PyDataFrame,
    inventory_contracts: Vec<String>,
    entry_filter_query: &str,
    contract_col: &str,
    cost_field: &str,
    entry_sort_col: Option<&str>,
    entry_sort_asc: bool,
    shares_per_contract: i64,
    is_sell: bool,
) -> PyResult<PyDataFrame> {
    let opts = py_to_polars(options_data);
    let compiled = filter::CompiledFilter::new(entry_filter_query)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let result = entries::compute_leg_entries(
        &opts,
        &inventory_contracts,
        &compiled,
        contract_col,
        cost_field,
        entry_sort_col,
        entry_sort_asc,
        shares_per_contract,
        is_sell,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(polars_to_py(result))
}
