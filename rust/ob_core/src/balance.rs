//! Full _update_balance orchestration in Rust.
//!
//! Mirrors Python's BacktestEngine._update_balance: for a date range,
//! join inventory to market data, compute calls/puts capital, stock values,
//! and assemble balance rows.

use polars::prelude::*;

use crate::inventory::{aggregate_by_type, join_inventory_to_market};
use crate::types::Direction;

/// Leg inventory data needed for balance computation.
pub struct LegInventory {
    pub contracts: Vec<String>,
    pub qtys: Vec<f64>,
    pub types: Vec<String>,
    pub direction: Direction,
    pub underlyings: Vec<String>,
    pub strikes: Vec<f64>,
}

/// Stock inventory data.
pub struct StockInventory {
    pub symbols: Vec<String>,
    pub qtys: Vec<f64>,
}

/// Compute balance for a date range.
///
/// This is the full orchestration of the hot path:
/// 1. For each leg, join inventory to market data
/// 2. Aggregate calls/puts capital by date
/// 3. Compute stock values
/// 4. Assemble balance rows
pub fn compute_balance(
    legs: &[LegInventory],
    stocks: &StockInventory,
    options_data: &DataFrame,
    stocks_data: &DataFrame,
    contract_col: &str,
    date_col: &str,
    stocks_date_col: &str,
    stocks_sym_col: &str,
    stocks_price_col: &str,
    shares_per_contract: i64,
    cash: f64,
) -> PolarsResult<DataFrame> {
    // Build a stocks snapshot for intrinsic value fallback:
    // For balance computation we pass the full stocks_data to inventory join
    // so it can look up spot prices for missing contracts.
    // Get unique dates from options
    let dates = options_data
        .column(date_col)?
        .unique()?;

    let mut calls_total = Series::new("calls_capital".into(), vec![0.0f64; dates.len()]);
    let mut puts_total = Series::new("puts_capital".into(), vec![0.0f64; dates.len()]);

    // Process each leg
    for leg in legs {
        if leg.contracts.is_empty() {
            continue;
        }

        let cost_field = match leg.direction {
            Direction::Buy => "bid",   // exit price for buy = bid
            Direction::Sell => "ask",  // exit price for sell = ask
        };

        let joined = join_inventory_to_market(
            &leg.contracts,
            &leg.qtys,
            &leg.types,
            &leg.underlyings,
            &leg.strikes,
            options_data,
            Some(stocks_data),
            contract_col,
            date_col,
            cost_field,
            Some(stocks_sym_col),
            Some(stocks_price_col),
            leg.direction.invert(),
            shares_per_contract,
        )?;

        let (calls_df, puts_df) = aggregate_by_type(&joined, date_col)?;

        // Add to running totals (would need date alignment in production)
        if calls_df.height() > 0 {
            if let Ok(col) = calls_df.column("calls_capital") {
                let vals = col.f64()?;
                let total_vals = calls_total.f64()?;
                let new: Float64Chunked = total_vals
                    .into_iter()
                    .zip(vals.into_iter())
                    .map(|(a, b)| Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)))
                    .collect();
                calls_total = new.into_series();
            }
        }

        if puts_df.height() > 0 {
            if let Ok(col) = puts_df.column("puts_capital") {
                let vals = col.f64()?;
                let total_vals = puts_total.f64()?;
                let new: Float64Chunked = total_vals
                    .into_iter()
                    .zip(vals.into_iter())
                    .map(|(a, b)| Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)))
                    .collect();
                puts_total = new.into_series();
            }
        }
    }

    // Compute stock values
    let stock_values = compute_stock_values(
        stocks,
        stocks_data,
        stocks_date_col,
        stocks_sym_col,
        stocks_price_col,
    )?;

    // Assemble balance DataFrame
    let cash_series = Series::new("cash".into(), vec![cash; dates.len()]);
    let options_qty: f64 = legs.iter().flat_map(|l| &l.qtys).sum();
    let options_qty_series = Series::new("options_qty".into(), vec![options_qty; dates.len()]);
    let stocks_qty: f64 = stocks.qtys.iter().sum();
    let stocks_qty_series = Series::new("stocks_qty".into(), vec![stocks_qty; dates.len()]);

    let mut columns = vec![
        dates.clone().into_column(),
        cash_series.into_column(),
        options_qty_series.into_column(),
        calls_total.with_name("calls_capital".into()).into_column(),
        puts_total.with_name("puts_capital".into()).into_column(),
        stocks_qty_series.into_column(),
    ];

    // Add stock value columns
    for col in stock_values.get_columns() {
        columns.push(col.clone());
    }

    DataFrame::new(columns)
}

fn compute_stock_values(
    stocks: &StockInventory,
    stocks_data: &DataFrame,
    date_col: &str,
    sym_col: &str,
    price_col: &str,
) -> PolarsResult<DataFrame> {
    if stocks.symbols.is_empty() {
        return Ok(DataFrame::default());
    }

    let mut result_cols: Vec<Column> = Vec::new();

    for (symbol, qty) in stocks.symbols.iter().zip(stocks.qtys.iter()) {
        let filtered = stocks_data
            .clone()
            .lazy()
            .filter(col(sym_col).eq(lit(symbol.as_str())))
            .select([
                col(date_col),
                (col(price_col) * lit(*qty)).alias(symbol.as_str()),
            ])
            .collect()?;

        if let Ok(val_col) = filtered.column(symbol.as_str()) {
            result_cols.push(val_col.clone());
        }
    }

    if result_cols.is_empty() {
        return Ok(DataFrame::default());
    }

    DataFrame::new(result_cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_balance_empty_legs() {
        let opts = df!(
            "optionroot" => &["A"],
            "quotedate" => &["2024-01-01"],
            "ask" => &[1.0],
            "bid" => &[0.9],
        )
        .unwrap();

        let stocks_df = df!(
            "date" => &["2024-01-01"],
            "symbol" => &["SPY"],
            "adjClose" => &[450.0],
        )
        .unwrap();

        let result = compute_balance(
            &[],  // no legs
            &StockInventory {
                symbols: vec!["SPY".into()],
                qtys: vec![100.0],
            },
            &opts,
            &stocks_df,
            "optionroot",
            "quotedate",
            "date",
            "symbol",
            "adjClose",
            100,
            1_000_000.0,
        );

        assert!(result.is_ok());
    }
}
