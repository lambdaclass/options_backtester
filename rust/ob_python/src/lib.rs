use pyo3::prelude::*;

mod arrow_bridge;
mod py_balance;
mod py_backtest;
mod py_convexity;
mod py_filter;
mod py_entries;
mod py_exits;
mod py_stats;
mod py_execution;
mod py_sweep;

#[pymodule]
fn _ob_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_balance::update_balance, m)?)?;
    m.add_function(wrap_pyfunction!(py_backtest::run_backtest_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_backtest::run_multi_strategy_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_filter::compile_filter, m)?)?;
    m.add_function(wrap_pyfunction!(py_filter::apply_filter, m)?)?;
    m.add_function(wrap_pyfunction!(py_entries::compute_entries, m)?)?;
    m.add_function(wrap_pyfunction!(py_exits::compute_exit_mask, m)?)?;
    m.add_function(wrap_pyfunction!(py_stats::compute_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_stats::compute_full_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_sweep::parallel_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(py_convexity::compute_daily_scores, m)?)?;
    m.add_function(wrap_pyfunction!(py_convexity::run_convexity_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution::rust_option_cost, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution::rust_stock_cost, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution::rust_fill_price, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution::rust_nearest_delta_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution::rust_max_value_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_execution::rust_risk_check, m)?)?;
    m.add_class::<py_filter::CompiledFilter>()?;
    Ok(())
}
