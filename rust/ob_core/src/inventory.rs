//! Inventory join — THE hot path.
//!
//! Mirrors the inner loop of Python's `_update_balance`:
//!   inv_info.merge(options_data, left_on="_contract", right_on=contract_col)
//!   then compute _value = sign * price * qty * shares_per_contract
//!   then groupby(date).sum() split by call/put.

use polars::prelude::*;

use crate::types::Direction;

/// Join inventory contracts with current market data and compute leg values.
///
/// Returns a DataFrame with columns: [date, _value, _type]
/// where _value = sign * price * qty * shares_per_contract.
pub fn join_inventory_to_market(
    contracts: &[String],
    qtys: &[f64],
    types: &[String],
    underlyings: &[String],
    strikes: &[f64],
    options_data: &DataFrame,
    stocks_data: Option<&DataFrame>,
    contract_col: &str,
    _date_col: &str,
    cost_field: &str,
    stocks_sym_col: Option<&str>,
    stocks_price_col: Option<&str>,
    direction: Direction,
    shares_per_contract: i64,
) -> PolarsResult<DataFrame> {
    let contract_series = Series::new("_contract".into(), contracts);
    let qty_series = Series::new("_qty".into(), qtys);
    let type_series = Series::new("_type".into(), types);
    let underlying_series = Series::new("_underlying".into(), underlyings);
    let strike_series = Series::new("_strike".into(), strikes);

    let inv = DataFrame::new(vec![
        contract_series.into_column(),
        qty_series.into_column(),
        type_series.into_column(),
        underlying_series.into_column(),
        strike_series.into_column(),
    ])?;

    let mut joined = inv
        .lazy()
        .join(
            options_data.clone().lazy(),
            [col("_contract")],
            [col(contract_col)],
            JoinArgs::new(JoinType::Left),
        )
        .collect()?;

    // Fill null cost fields with intrinsic value from stocks data
    if let Some(cost_col) = joined.column(cost_field).ok() {
        let null_mask = cost_col.is_null();
        if null_mask.sum().unwrap_or(0) > 0 {
            // Build a price lookup from stocks data (latest price per symbol)
            let mut price_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
            if let (Some(sdf), Some(sym_c), Some(price_c)) = (stocks_data, stocks_sym_col, stocks_price_col) {
                if let (Ok(sym_ca), Ok(price_raw)) = (sdf.column(sym_c), sdf.column(price_c)) {
                    if let Ok(sym_str) = sym_ca.str() {
                        let price_casted = price_raw.cast(&DataType::Float64).unwrap_or(price_raw.clone());
                        if let Ok(price_ca) = price_casted.f64() {
                            for i in 0..sdf.height() {
                                if let (Some(s), Some(p)) = (sym_str.get(i), price_ca.get(i)) {
                                    price_map.insert(s.to_string(), p);
                                }
                            }
                        }
                    }
                }
            }

            let types_ca = joined.column("_type")?.str()?;
            let strikes_ca = joined.column("_strike")?.f64()?;
            let underlyings_ca = joined.column("_underlying")?.str()?;
            let cost_ca = joined.column(cost_field)?.f64()?;

            let filled: Vec<Option<f64>> = (0..joined.height())
                .map(|i| {
                    if cost_ca.get(i).is_some() {
                        cost_ca.get(i)
                    } else {
                        let opt_type = types_ca.get(i).unwrap_or("put");
                        let strike = strikes_ca.get(i).unwrap_or(0.0);
                        let underlying = underlyings_ca.get(i).unwrap_or("");
                        let spot = price_map.get(underlying).copied().unwrap_or(0.0);
                        let iv = if opt_type == "call" {
                            (spot - strike).max(0.0)
                        } else {
                            (strike - spot).max(0.0)
                        };
                        Some(iv)
                    }
                })
                .collect();

            let filled_series = Float64Chunked::from_iter_options(cost_field.into(), filled.into_iter());
            let _ = joined.replace(cost_field, filled_series.into_series());
        }
    }

    // Compute _value after filling nulls
    let joined = joined
        .lazy()
        .with_column(
            (lit(direction.sign())
                * col(cost_field)
                * col("_qty")
                * lit(shares_per_contract as f64))
            .alias("_value"),
        )
        .collect()?;

    Ok(joined)
}

/// Aggregate values by date, split into calls and puts capital.
///
/// Returns (calls_by_date, puts_by_date) as Series indexed by date.
pub fn aggregate_by_type(
    joined: &DataFrame,
    date_col: &str,
) -> PolarsResult<(DataFrame, DataFrame)> {
    let calls = joined
        .clone()
        .lazy()
        .filter(col("_type").eq(lit("call")))
        .group_by([col(date_col)])
        .agg([col("_value").sum().alias("calls_capital")])
        .sort([date_col], Default::default())
        .collect()?;

    let puts = joined
        .clone()
        .lazy()
        .filter(col("_type").neq(lit("call")))
        .group_by([col(date_col)])
        .agg([col("_value").sum().alias("puts_capital")])
        .sort([date_col], Default::default())
        .collect()?;

    Ok((calls, puts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_options() -> DataFrame {
        df!(
            "optionroot" => &["SPX_A", "SPX_A", "SPX_B", "SPX_B"],
            "quotedate" => &["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "ask" => &[2.0, 2.5, 3.0, 3.5],
            "bid" => &[1.8, 2.3, 2.8, 3.3],
        )
        .unwrap()
    }

    #[test]
    fn join_computes_values() {
        let opts = sample_options();
        let result = join_inventory_to_market(
            &["SPX_A".into(), "SPX_B".into()],
            &[10.0, 5.0],
            &["call".into(), "put".into()],
            &["SPY".into(), "SPY".into()],
            &[400.0, 400.0],
            &opts,
            None,
            "optionroot",
            "quotedate",
            "bid",
            None,
            None,
            Direction::Buy,
            100,
        )
        .unwrap();

        assert!(result.height() > 0);
        let values = result.column("_value").unwrap();
        // Direction::Buy sign = -1, so values should be negative
        let first_val: f64 = values.f64().unwrap().get(0).unwrap();
        assert!(first_val < 0.0);
    }

    #[test]
    fn aggregate_splits_calls_puts() {
        let opts = sample_options();
        let joined = join_inventory_to_market(
            &["SPX_A".into(), "SPX_B".into()],
            &[10.0, 5.0],
            &["call".into(), "put".into()],
            &["SPY".into(), "SPY".into()],
            &[400.0, 400.0],
            &opts,
            None,
            "optionroot",
            "quotedate",
            "bid",
            None,
            None,
            Direction::Buy,
            100,
        )
        .unwrap();

        let (calls, puts) = aggregate_by_type(&joined, "quotedate").unwrap();
        assert!(calls.height() > 0);
        assert!(puts.height() > 0);
    }
}
