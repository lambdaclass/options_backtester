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
    options_data: &DataFrame,
    contract_col: &str,
    _date_col: &str,
    cost_field: &str,
    direction: Direction,
    shares_per_contract: i64,
) -> PolarsResult<DataFrame> {
    let contract_series = Series::new("_contract".into(), contracts);
    let qty_series = Series::new("_qty".into(), qtys);
    let type_series = Series::new("_type".into(), types);

    let inv = DataFrame::new(vec![
        contract_series.into_column(),
        qty_series.into_column(),
        type_series.into_column(),
    ])?;

    let joined = inv
        .lazy()
        .join(
            options_data.clone().lazy(),
            [col("_contract")],
            [col(contract_col)],
            JoinArgs::new(JoinType::Left),
        )
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
            &opts,
            "optionroot",
            "quotedate",
            "bid",
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
            &opts,
            "optionroot",
            "quotedate",
            "bid",
            Direction::Buy,
            100,
        )
        .unwrap();

        let (calls, puts) = aggregate_by_type(&joined, "quotedate").unwrap();
        assert!(calls.height() > 0);
        assert!(puts.height() > 0);
    }
}
