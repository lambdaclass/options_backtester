/// Arrow C Data Interface bridge: pyarrow <-> Polars zero-copy.
///
/// Uses pyo3-polars for direct DataFrame conversions between
/// Python (pandas/pyarrow) and Rust (Polars).

use pyo3_polars::PyDataFrame;
use polars::prelude::DataFrame;

/// Convert a PyDataFrame (from Python) to a Polars DataFrame.
pub fn py_to_polars(py_df: PyDataFrame) -> DataFrame {
    py_df.0
}

/// Convert a Polars DataFrame to a PyDataFrame (for Python).
pub fn polars_to_py(df: DataFrame) -> PyDataFrame {
    PyDataFrame(df)
}
