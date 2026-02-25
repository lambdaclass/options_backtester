//! PyO3 bindings for stats computation.

use pyo3::prelude::*;

use ob_core::stats;

/// Compute backtest statistics from daily returns and trade PnLs.
#[pyfunction]
#[pyo3(signature = (daily_returns, trade_pnls, risk_free_rate = 0.0))]
pub fn compute_stats(
    daily_returns: Vec<f64>,
    trade_pnls: Vec<f64>,
    risk_free_rate: f64,
) -> PyResult<PyObject> {
    let s = stats::compute_stats(&daily_returns, &trade_pnls, risk_free_rate);

    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_return", s.total_return)?;
        dict.set_item("annualized_return", s.annualized_return)?;
        dict.set_item("sharpe_ratio", s.sharpe_ratio)?;
        dict.set_item("sortino_ratio", s.sortino_ratio)?;
        dict.set_item("calmar_ratio", s.calmar_ratio)?;
        dict.set_item("max_drawdown", s.max_drawdown)?;
        dict.set_item("max_drawdown_duration", s.max_drawdown_duration)?;
        dict.set_item("profit_factor", s.profit_factor)?;
        dict.set_item("win_rate", s.win_rate)?;
        dict.set_item("total_trades", s.total_trades)?;
        Ok(dict.into())
    })
}
