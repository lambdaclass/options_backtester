//! PyO3 bindings for filter compilation and evaluation.

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use ob_core::filter;

use crate::arrow_bridge::{polars_to_py, py_to_polars};

/// Compiled filter that can be reused across multiple evaluations.
#[pyclass]
pub struct CompiledFilter {
    inner: filter::CompiledFilter,
}

#[pymethods]
impl CompiledFilter {
    #[new]
    fn new(query: &str) -> PyResult<Self> {
        let inner = filter::CompiledFilter::new(query)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn apply(&self, data: PyDataFrame) -> PyResult<PyDataFrame> {
        let df = py_to_polars(data);
        let result = self
            .inner
            .apply(&df)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(polars_to_py(result))
    }

    fn __repr__(&self) -> String {
        format!("CompiledFilter({:?})", self.inner.expr)
    }
}

/// Compile a filter query string and return a CompiledFilter.
#[pyfunction]
pub fn compile_filter(query: &str) -> PyResult<CompiledFilter> {
    CompiledFilter::new(query)
}

/// One-shot: compile and apply a filter in one call.
#[pyfunction]
pub fn apply_filter(query: &str, data: PyDataFrame) -> PyResult<PyDataFrame> {
    let f = filter::CompiledFilter::new(query)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let df = py_to_polars(data);
    let result = f
        .apply(&df)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(polars_to_py(result))
}
