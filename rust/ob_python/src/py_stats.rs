//! PyO3 bindings for stats computation.

use pyo3::prelude::*;

use ob_core::stats;

/// Compute backtest statistics from daily returns and trade PnLs (legacy).
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

/// Compute comprehensive backtest statistics from total capital series.
///
/// Args:
///     total_capital: list of daily total capital values
///     timestamps_ns: list of nanosecond timestamps (one per capital value)
///     trade_pnls: list of per-trade P&L values
///     stock_weights: flattened [n_days × n_stocks] weight matrix (row-major)
///     n_stocks: number of stock columns
///     risk_free_rate: annualized risk-free rate (default 0.0)
#[pyfunction]
#[pyo3(signature = (total_capital, timestamps_ns, trade_pnls, stock_weights, n_stocks, risk_free_rate = 0.0))]
pub fn compute_full_stats(
    total_capital: Vec<f64>,
    timestamps_ns: Vec<i64>,
    trade_pnls: Vec<f64>,
    stock_weights: Vec<f64>,
    n_stocks: usize,
    risk_free_rate: f64,
) -> PyResult<PyObject> {
    let fs = stats::compute_full_stats(
        &total_capital,
        &timestamps_ns,
        &trade_pnls,
        &stock_weights,
        n_stocks,
        risk_free_rate,
    );

    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);

        // Trade stats
        dict.set_item("total_trades", fs.total_trades)?;
        dict.set_item("wins", fs.wins)?;
        dict.set_item("losses", fs.losses)?;
        dict.set_item("win_pct", fs.win_pct)?;
        dict.set_item("profit_factor", fs.profit_factor)?;
        dict.set_item("largest_win", fs.largest_win)?;
        dict.set_item("largest_loss", fs.largest_loss)?;
        dict.set_item("avg_win", fs.avg_win)?;
        dict.set_item("avg_loss", fs.avg_loss)?;
        dict.set_item("avg_trade", fs.avg_trade)?;

        // Return stats
        dict.set_item("total_return", fs.total_return)?;
        dict.set_item("annualized_return", fs.annualized_return)?;
        dict.set_item("sharpe_ratio", fs.sharpe_ratio)?;
        dict.set_item("sortino_ratio", fs.sortino_ratio)?;
        dict.set_item("calmar_ratio", fs.calmar_ratio)?;

        // Risk stats
        dict.set_item("max_drawdown", fs.max_drawdown)?;
        dict.set_item("max_drawdown_duration", fs.max_drawdown_duration)?;
        dict.set_item("avg_drawdown", fs.avg_drawdown)?;
        dict.set_item("avg_drawdown_duration", fs.avg_drawdown_duration)?;
        dict.set_item("volatility", fs.volatility)?;
        dict.set_item("tail_ratio", fs.tail_ratio)?;

        // Daily period stats
        let daily = pyo3::types::PyDict::new(py);
        daily.set_item("mean", fs.daily.mean)?;
        daily.set_item("vol", fs.daily.vol)?;
        daily.set_item("sharpe", fs.daily.sharpe)?;
        daily.set_item("sortino", fs.daily.sortino)?;
        daily.set_item("skew", fs.daily.skew)?;
        daily.set_item("kurtosis", fs.daily.kurtosis)?;
        daily.set_item("best", fs.daily.best)?;
        daily.set_item("worst", fs.daily.worst)?;
        dict.set_item("daily", daily)?;

        // Monthly period stats
        let monthly = pyo3::types::PyDict::new(py);
        monthly.set_item("mean", fs.monthly.mean)?;
        monthly.set_item("vol", fs.monthly.vol)?;
        monthly.set_item("sharpe", fs.monthly.sharpe)?;
        monthly.set_item("sortino", fs.monthly.sortino)?;
        monthly.set_item("skew", fs.monthly.skew)?;
        monthly.set_item("kurtosis", fs.monthly.kurtosis)?;
        monthly.set_item("best", fs.monthly.best)?;
        monthly.set_item("worst", fs.monthly.worst)?;
        dict.set_item("monthly", monthly)?;

        // Yearly period stats
        let yearly = pyo3::types::PyDict::new(py);
        yearly.set_item("mean", fs.yearly.mean)?;
        yearly.set_item("vol", fs.yearly.vol)?;
        yearly.set_item("sharpe", fs.yearly.sharpe)?;
        yearly.set_item("sortino", fs.yearly.sortino)?;
        yearly.set_item("skew", fs.yearly.skew)?;
        yearly.set_item("kurtosis", fs.yearly.kurtosis)?;
        yearly.set_item("best", fs.yearly.best)?;
        yearly.set_item("worst", fs.yearly.worst)?;
        dict.set_item("yearly", yearly)?;

        // Lookback returns
        let lookback = pyo3::types::PyDict::new(py);
        set_opt(&lookback, "mtd", fs.lookback.mtd)?;
        set_opt(&lookback, "three_month", fs.lookback.three_month)?;
        set_opt(&lookback, "six_month", fs.lookback.six_month)?;
        set_opt(&lookback, "ytd", fs.lookback.ytd)?;
        set_opt(&lookback, "one_year", fs.lookback.one_year)?;
        set_opt(&lookback, "three_year", fs.lookback.three_year)?;
        set_opt(&lookback, "five_year", fs.lookback.five_year)?;
        set_opt(&lookback, "ten_year", fs.lookback.ten_year)?;
        dict.set_item("lookback", lookback)?;

        // Portfolio metrics
        dict.set_item("turnover", fs.turnover)?;
        dict.set_item("herfindahl", fs.herfindahl)?;

        Ok(dict.into())
    })
}

fn set_opt(dict: &Bound<'_, pyo3::types::PyDict>, key: &str, val: Option<f64>) -> PyResult<()> {
    match val {
        Some(v) => dict.set_item(key, v)?,
        None => dict.set_item(key, pyo3::types::PyNone::get(dict.py()))?,
    }
    Ok(())
}
