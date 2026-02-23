/// PyO3 bindings for full backtest loop.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use ob_core::backtest::{run_backtest, BacktestConfig, SchemaMapping};
use ob_core::types::{Direction, LegConfig, OptionType};

use crate::arrow_bridge::{polars_to_py, py_to_polars};

/// Parse schema dict → SchemaMapping.
pub fn parse_schema(schema: &Bound<'_, PyDict>) -> PyResult<SchemaMapping> {
    Ok(SchemaMapping {
        contract: get_str(schema, "contract", "optionroot")?,
        date: get_str(schema, "date", "quotedate")?,
        stocks_date: get_str(schema, "stocks_date", "date")?,
        stocks_sym: get_str(schema, "stocks_symbol", "symbol")?,
        stocks_price: get_str(schema, "stocks_price", "adjClose")?,
        underlying: get_str(schema, "underlying", "underlying")?,
        expiration: get_str(schema, "expiration", "expiration")?,
        option_type: get_str(schema, "type", "type")?,
        strike: get_str(schema, "strike", "strike")?,
    })
}

/// Parse config dict → BacktestConfig.
pub fn parse_config_from_dict(config: &Bound<'_, PyDict>) -> PyResult<BacktestConfig> {
    let alloc_obj = config
        .get_item("allocation")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("allocation"))?;
    let alloc: &Bound<'_, PyDict> = alloc_obj
        .downcast::<PyDict>()
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;

    let alloc_stocks = get_f64(alloc, "stocks", 0.0)?;
    let alloc_options = get_f64(alloc, "options", 0.0)?;
    let alloc_cash = get_f64(alloc, "cash", 0.0)?;

    let initial_capital = get_f64(config, "initial_capital", 1_000_000.0)?;
    let spc = get_i64(config, "shares_per_contract", 100)?;

    let profit_pct: Option<f64> = config
        .get_item("profit_pct")?
        .and_then(|v| v.extract::<f64>().ok());
    let loss_pct: Option<f64> = config
        .get_item("loss_pct")?
        .and_then(|v| v.extract::<f64>().ok());

    let rebalance_dates: Vec<String> = config
        .get_item("rebalance_dates")?
        .map(|v| v.extract::<Vec<String>>())
        .transpose()?
        .unwrap_or_default();

    let legs_list: Vec<Bound<'_, PyDict>> = config
        .get_item("legs")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("legs"))?
        .extract::<Vec<Bound<'_, PyDict>>>()?;

    let legs: Vec<LegConfig> = legs_list
        .iter()
        .map(|d| parse_leg_config(d))
        .collect::<PyResult<Vec<_>>>()?;

    let stocks_list: Vec<(String, f64)> = config
        .get_item("stocks")?
        .map(|v| v.extract::<Vec<(String, f64)>>())
        .transpose()?
        .unwrap_or_default();

    let stock_symbols: Vec<String> = stocks_list.iter().map(|(s, _)| s.clone()).collect();
    let stock_percentages: Vec<f64> = stocks_list.iter().map(|(_, p)| *p).collect();

    Ok(BacktestConfig {
        allocation_stocks: alloc_stocks,
        allocation_options: alloc_options,
        allocation_cash: alloc_cash,
        initial_capital,
        shares_per_contract: spc,
        legs,
        profit_pct,
        loss_pct,
        stock_symbols,
        stock_percentages,
        rebalance_dates,
    })
}

/// Run a full backtest and return (balance_df, trade_log_df, stats_dict).
#[pyfunction]
#[pyo3(signature = (options_data, stocks_data, config, schema_mapping))]
pub fn run_backtest_py(
    py: Python<'_>,
    options_data: PyDataFrame,
    stocks_data: PyDataFrame,
    config: &Bound<'_, PyDict>,
    schema_mapping: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let opts = py_to_polars(options_data);
    let stocks = py_to_polars(stocks_data);

    let schema = parse_schema(schema_mapping)?;
    let bt_config = parse_config_from_dict(config)?;

    let result = run_backtest(&bt_config, &opts, &stocks, &schema)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Build result tuple
    let balance_py = polars_to_py(result.balance);
    let trade_log_py = polars_to_py(result.trade_log);

    let stats_dict = PyDict::new(py);
    stats_dict.set_item("total_return", result.stats.total_return)?;
    stats_dict.set_item("annualized_return", result.stats.annualized_return)?;
    stats_dict.set_item("sharpe_ratio", result.stats.sharpe_ratio)?;
    stats_dict.set_item("sortino_ratio", result.stats.sortino_ratio)?;
    stats_dict.set_item("calmar_ratio", result.stats.calmar_ratio)?;
    stats_dict.set_item("max_drawdown", result.stats.max_drawdown)?;
    stats_dict.set_item("max_drawdown_duration", result.stats.max_drawdown_duration)?;
    stats_dict.set_item("profit_factor", result.stats.profit_factor)?;
    stats_dict.set_item("win_rate", result.stats.win_rate)?;
    stats_dict.set_item("total_trades", result.stats.total_trades)?;
    stats_dict.set_item("final_cash", result.final_cash)?;

    let result_tuple = pyo3::types::PyTuple::new(py, [
        balance_py.into_pyobject(py)?.into_any(),
        trade_log_py.into_pyobject(py)?.into_any(),
        stats_dict.into_any(),
    ])?;

    Ok(result_tuple.into())
}

pub fn parse_leg_config(d: &Bound<'_, PyDict>) -> PyResult<LegConfig> {
    let name = get_str(d, "name", "")?;
    let direction_str = get_str(d, "direction", "ask")?;
    let type_str = get_str(d, "type", "call")?;
    let entry_filter: Option<String> = d.get_item("entry_filter")?.and_then(|v| v.extract().ok());
    let exit_filter: Option<String> = d.get_item("exit_filter")?.and_then(|v| v.extract().ok());
    let entry_sort_col: Option<String> = d.get_item("entry_sort_col")?.and_then(|v| v.extract().ok());
    let entry_sort_asc: bool = d.get_item("entry_sort_asc")?
        .map(|v| v.extract::<bool>()).transpose()?.unwrap_or(true);

    Ok(LegConfig {
        name,
        option_type: if type_str == "put" { OptionType::Put } else { OptionType::Call },
        direction: if direction_str == "bid" { Direction::Sell } else { Direction::Buy },
        entry_filter_query: entry_filter,
        exit_filter_query: exit_filter,
        entry_sort_col,
        entry_sort_asc,
    })
}

// Helper extractors
pub fn get_str(d: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    Ok(d.get_item(key)?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_else(|| default.into()))
}
pub fn get_f64(d: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    Ok(d.get_item(key)?.map(|v| v.extract::<f64>()).transpose()?.unwrap_or(default))
}
pub fn get_i64(d: &Bound<'_, PyDict>, key: &str, default: i64) -> PyResult<i64> {
    Ok(d.get_item(key)?.map(|v| v.extract::<i64>()).transpose()?.unwrap_or(default))
}
