use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ob_core::convexity_scoring;
use ob_core::convexity_backtest;

#[pyfunction]
#[pyo3(signature = (
    dates_ns, strikes, bids, asks, deltas, underlying_prices, dtes, implied_vols,
    target_delta, dte_min, dte_max, tail_drop
))]
pub fn compute_daily_scores<'py>(
    py: Python<'py>,
    dates_ns: PyReadonlyArray1<'py, i64>,
    strikes: PyReadonlyArray1<'py, f64>,
    bids: PyReadonlyArray1<'py, f64>,
    asks: PyReadonlyArray1<'py, f64>,
    deltas: PyReadonlyArray1<'py, f64>,
    underlying_prices: PyReadonlyArray1<'py, f64>,
    dtes: PyReadonlyArray1<'py, i32>,
    implied_vols: PyReadonlyArray1<'py, f64>,
    target_delta: f64,
    dte_min: i32,
    dte_max: i32,
    tail_drop: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let scores = convexity_scoring::compute_daily_scores(
        dates_ns.as_slice()?,
        strikes.as_slice()?,
        bids.as_slice()?,
        asks.as_slice()?,
        deltas.as_slice()?,
        underlying_prices.as_slice()?,
        dtes.as_slice()?,
        implied_vols.as_slice()?,
        target_delta,
        dte_min,
        dte_max,
        tail_drop,
    );

    let dict = PyDict::new(py);
    dict.set_item("dates_ns", scores.iter().map(|s| s.date_ns).collect::<Vec<_>>())?;
    dict.set_item("convexity_ratios", scores.iter().map(|s| s.convexity_ratio).collect::<Vec<_>>())?;
    dict.set_item("strikes", scores.iter().map(|s| s.strike).collect::<Vec<_>>())?;
    dict.set_item("asks", scores.iter().map(|s| s.ask).collect::<Vec<_>>())?;
    dict.set_item("bids", scores.iter().map(|s| s.bid).collect::<Vec<_>>())?;
    dict.set_item("deltas", scores.iter().map(|s| s.delta).collect::<Vec<_>>())?;
    dict.set_item("underlying_prices", scores.iter().map(|s| s.underlying_price).collect::<Vec<_>>())?;
    dict.set_item("implied_vols", scores.iter().map(|s| s.implied_vol).collect::<Vec<_>>())?;
    dict.set_item("dtes", scores.iter().map(|s| s.dte).collect::<Vec<_>>())?;
    dict.set_item("annual_costs", scores.iter().map(|s| s.annual_cost).collect::<Vec<_>>())?;
    dict.set_item("tail_payoffs", scores.iter().map(|s| s.tail_payoff).collect::<Vec<_>>())?;

    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (
    put_dates_ns, put_expirations_ns, put_strikes, put_bids, put_asks,
    put_deltas, put_underlying, put_dtes, put_ivs,
    stock_dates_ns, stock_prices,
    initial_capital, budget_pct, target_delta, dte_min, dte_max, tail_drop
))]
#[allow(clippy::too_many_arguments)]
pub fn run_convexity_backtest<'py>(
    py: Python<'py>,
    put_dates_ns: PyReadonlyArray1<'py, i64>,
    put_expirations_ns: PyReadonlyArray1<'py, i64>,
    put_strikes: PyReadonlyArray1<'py, f64>,
    put_bids: PyReadonlyArray1<'py, f64>,
    put_asks: PyReadonlyArray1<'py, f64>,
    put_deltas: PyReadonlyArray1<'py, f64>,
    put_underlying: PyReadonlyArray1<'py, f64>,
    put_dtes: PyReadonlyArray1<'py, i32>,
    put_ivs: PyReadonlyArray1<'py, f64>,
    stock_dates_ns: PyReadonlyArray1<'py, i64>,
    stock_prices: PyReadonlyArray1<'py, f64>,
    initial_capital: f64,
    budget_pct: f64,
    target_delta: f64,
    dte_min: i32,
    dte_max: i32,
    tail_drop: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = convexity_backtest::run_backtest(
        put_dates_ns.as_slice()?,
        put_expirations_ns.as_slice()?,
        put_strikes.as_slice()?,
        put_bids.as_slice()?,
        put_asks.as_slice()?,
        put_deltas.as_slice()?,
        put_underlying.as_slice()?,
        put_dtes.as_slice()?,
        put_ivs.as_slice()?,
        stock_dates_ns.as_slice()?,
        stock_prices.as_slice()?,
        initial_capital,
        budget_pct,
        target_delta,
        dte_min,
        dte_max,
        tail_drop,
    );

    let dict = PyDict::new(py);

    // Monthly records
    let records = PyDict::new(py);
    records.set_item("dates_ns", result.records.iter().map(|r| r.date_ns).collect::<Vec<_>>())?;
    records.set_item("shares", result.records.iter().map(|r| r.shares).collect::<Vec<_>>())?;
    records.set_item("stock_prices", result.records.iter().map(|r| r.stock_price).collect::<Vec<_>>())?;
    records.set_item("equity_values", result.records.iter().map(|r| r.equity_value).collect::<Vec<_>>())?;
    records.set_item("put_costs", result.records.iter().map(|r| r.put_cost).collect::<Vec<_>>())?;
    records.set_item("put_exit_values", result.records.iter().map(|r| r.put_exit_value).collect::<Vec<_>>())?;
    records.set_item("put_pnls", result.records.iter().map(|r| r.put_pnl).collect::<Vec<_>>())?;
    records.set_item("portfolio_values", result.records.iter().map(|r| r.portfolio_value).collect::<Vec<_>>())?;
    records.set_item("convexity_ratios", result.records.iter().map(|r| r.convexity_ratio).collect::<Vec<_>>())?;
    records.set_item("strikes", result.records.iter().map(|r| r.strike).collect::<Vec<_>>())?;
    records.set_item("contracts", result.records.iter().map(|r| r.contracts).collect::<Vec<_>>())?;
    dict.set_item("records", records)?;

    // Daily balance series
    dict.set_item("daily_dates_ns", result.daily_dates_ns)?;
    dict.set_item("daily_balances", result.daily_balances)?;

    Ok(dict)
}
