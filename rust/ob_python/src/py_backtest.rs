/// PyO3 bindings for full backtest loop.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

use ob_core::backtest::{run_backtest, BacktestConfig, SchemaMapping};
use ob_core::types::{Direction, LegConfig, OptionType};

use crate::arrow_bridge::{polars_to_py, py_to_polars};

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

    // Parse schema mapping
    let contract_col = get_str(schema_mapping, "contract", "optionroot")?;
    let date_col = get_str(schema_mapping, "date", "quotedate")?;
    let stocks_date_col = get_str(schema_mapping, "stocks_date", "date")?;
    let stocks_sym_col = get_str(schema_mapping, "stocks_symbol", "symbol")?;
    let stocks_price_col = get_str(schema_mapping, "stocks_price", "adjClose")?;
    let underlying_col = get_str(schema_mapping, "underlying", "underlying")?;
    let expiration_col = get_str(schema_mapping, "expiration", "expiration")?;
    let type_col = get_str(schema_mapping, "type", "type")?;
    let strike_col = get_str(schema_mapping, "strike", "strike")?;

    let schema = SchemaMapping {
        underlying: underlying_col,
        expiration: expiration_col,
        option_type: type_col,
        strike: strike_col,
    };

    // Parse allocation
    let alloc_obj = config
        .get_item("allocation")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("allocation"))?;
    let alloc: &Bound<'_, PyDict> = alloc_obj
        .downcast::<PyDict>()
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;

    let alloc_stocks = get_f64_from(alloc, "stocks", 0.0)?;
    let alloc_options = get_f64_from(alloc, "options", 0.0)?;
    let alloc_cash = get_f64_from(alloc, "cash", 0.0)?;

    let initial_capital = get_f64(config, "initial_capital", 1_000_000.0)?;
    let spc = get_i64(config, "shares_per_contract", 100)?;

    let profit_pct: Option<f64> = config
        .get_item("profit_pct")?
        .and_then(|v| v.extract::<f64>().ok());
    let loss_pct: Option<f64> = config
        .get_item("loss_pct")?
        .and_then(|v| v.extract::<f64>().ok());

    // Parse rebalance dates (pre-computed in Python)
    let rebalance_dates: Vec<String> = config
        .get_item("rebalance_dates")?
        .map(|v| v.extract::<Vec<String>>())
        .transpose()?
        .unwrap_or_default();

    // Parse legs
    let legs_list: Vec<Bound<'_, PyDict>> = config
        .get_item("legs")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("legs"))?
        .extract::<Vec<Bound<'_, PyDict>>>()?;

    let legs: Vec<LegConfig> = legs_list
        .iter()
        .map(|d| parse_leg_config(d))
        .collect::<PyResult<Vec<_>>>()?;

    // Parse stocks
    let stocks_list: Vec<(String, f64)> = config
        .get_item("stocks")?
        .map(|v| v.extract::<Vec<(String, f64)>>())
        .transpose()?
        .unwrap_or_default();

    let stock_symbols: Vec<String> = stocks_list.iter().map(|(s, _)| s.clone()).collect();
    let stock_percentages: Vec<f64> = stocks_list.iter().map(|(_, p)| *p).collect();

    let bt_config = BacktestConfig {
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
    };

    let result = run_backtest(
        &bt_config, &opts, &stocks,
        &contract_col, &date_col,
        &stocks_date_col, &stocks_sym_col, &stocks_price_col,
        &schema,
    )
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

fn parse_leg_config(d: &Bound<'_, PyDict>) -> PyResult<LegConfig> {
    let name = get_str_from(d, "name", "")?;
    let direction_str = get_str_from(d, "direction", "ask")?;
    let type_str = get_str_from(d, "type", "call")?;
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
fn get_str(d: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    Ok(d.get_item(key)?.map(|v| v.extract::<String>()).transpose()?.unwrap_or_else(|| default.into()))
}
fn get_str_from(d: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> { get_str(d, key, default) }
fn get_f64(d: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    Ok(d.get_item(key)?.map(|v| v.extract::<f64>()).transpose()?.unwrap_or(default))
}
fn get_f64_from(d: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> { get_f64(d, key, default) }
fn get_i64(d: &Bound<'_, PyDict>, key: &str, default: i64) -> PyResult<i64> {
    Ok(d.get_item(key)?.map(|v| v.extract::<i64>()).transpose()?.unwrap_or(default))
}
