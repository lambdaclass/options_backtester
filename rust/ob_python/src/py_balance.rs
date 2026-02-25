//! PyO3 bindings for balance update.

use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use ob_core::balance::{compute_balance, LegInventory, StockInventory};
use ob_core::types::Direction;

use crate::arrow_bridge::{polars_to_py, py_to_polars};

#[pyfunction]
#[pyo3(signature = (
    leg_contracts, leg_qtys, leg_types, leg_directions,
    stock_symbols, stock_qtys,
    options_data, stocks_data,
    contract_col, date_col,
    stocks_date_col, stocks_sym_col, stocks_price_col,
    shares_per_contract, cash,
))]
pub fn update_balance(
    leg_contracts: Vec<Vec<String>>,
    leg_qtys: Vec<Vec<f64>>,
    leg_types: Vec<Vec<String>>,
    leg_directions: Vec<String>,
    stock_symbols: Vec<String>,
    stock_qtys: Vec<f64>,
    options_data: PyDataFrame,
    stocks_data: PyDataFrame,
    contract_col: &str,
    date_col: &str,
    stocks_date_col: &str,
    stocks_sym_col: &str,
    stocks_price_col: &str,
    shares_per_contract: i64,
    cash: f64,
) -> PyResult<PyDataFrame> {
    let opts_df = py_to_polars(options_data);
    let stocks_df = py_to_polars(stocks_data);

    let legs: Vec<LegInventory> = leg_contracts
        .into_iter()
        .zip(leg_qtys)
        .zip(leg_types)
        .zip(leg_directions)
        .map(|(((contracts, qtys), types), dir)| LegInventory {
            contracts,
            qtys,
            types,
            direction: if dir == "buy" { Direction::Buy } else { Direction::Sell },
        })
        .collect();

    let stocks = StockInventory {
        symbols: stock_symbols,
        qtys: stock_qtys,
    };

    let result = compute_balance(
        &legs,
        &stocks,
        &opts_df,
        &stocks_df,
        contract_col,
        date_col,
        stocks_date_col,
        stocks_sym_col,
        stocks_price_col,
        shares_per_contract,
        cash,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(polars_to_py(result))
}
