//! Entry signal computation in Rust.
//!
//! Mirrors Python's _execute_option_entries:
//! 1. Anti-join to exclude held contracts
//! 2. Apply entry filter
//! 3. Sort by entry_sort
//! 4. Select signal fields
//! 5. Compute totals (cost, qty)

use polars::prelude::*;

use crate::filter::CompiledFilter;

/// Compute entry candidates for a single leg.
///
/// Steps:
/// 1. Anti-join options with inventory contracts
/// 2. Apply compiled entry filter
/// 3. Sort if entry_sort specified
/// 4. Select and rename signal fields
pub fn compute_leg_entries(
    options: &DataFrame,
    inventory_contracts: &[String],
    entry_filter: &CompiledFilter,
    contract_col: &str,
    cost_field: &str,
    entry_sort_col: Option<&str>,
    entry_sort_asc: bool,
    shares_per_contract: i64,
    is_sell: bool,
) -> PolarsResult<DataFrame> {
    // Anti-join: exclude already-held contracts
    let inv_contracts = Series::new("_held".into(), inventory_contracts);
    let inv_df = DataFrame::new(vec![inv_contracts.into_column()])?;

    let mut lazy = options
        .clone()
        .lazy()
        .join(
            inv_df.lazy(),
            [col(contract_col)],
            [col("_held")],
            JoinArgs::new(JoinType::Anti),
        );

    // Apply entry filter
    lazy = lazy.filter(entry_filter.polars_expr.clone());

    // Sort if specified
    if let Some(sort_col) = entry_sort_col {
        lazy = lazy.sort(
            [sort_col],
            SortMultipleOptions::default().with_order_descending(!entry_sort_asc),
        );
    }

    // Select signal fields and compute cost
    let sign = if is_sell { lit(-1.0) } else { lit(1.0) };
    let spc = lit(shares_per_contract as f64);

    lazy = lazy.select([
        col(contract_col).alias("contract"),
        col("underlying"),
        col("expiration"),
        col("type"),
        col("strike"),
        (sign * col(cost_field) * spc).alias("cost"),
    ]);

    lazy.collect()
}

/// Compute entry quantities given total costs and available allocation.
pub fn compute_entry_qty(total_costs: &Series, allocation: f64) -> PolarsResult<Series> {
    let abs_costs = total_costs.f64()?.apply(|v| v.map(|x| x.abs()));
    let qty: Float64Chunked = abs_costs
        .into_iter()
        .map(|c| c.map(|cost| if cost > 0.0 { (allocation / cost).floor() } else { 0.0 }))
        .collect();
    Ok(qty.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_options() -> DataFrame {
        df!(
            "optionroot" => &["A", "B", "C", "D"],
            "underlying" => &["SPX", "SPX", "SPX", "SPX"],
            "type" => &["put", "put", "call", "put"],
            "expiration" => &["2024-06-01", "2024-06-01", "2024-06-01", "2024-06-01"],
            "strike" => &[4000.0, 4100.0, 4200.0, 4300.0],
            "ask" => &[10.0, 15.0, 20.0, 25.0],
            "bid" => &[9.0, 14.0, 19.0, 24.0],
            "dte" => &[90i64, 90, 90, 90],
        )
        .unwrap()
    }

    #[test]
    fn compute_entries_excludes_held() {
        let opts = sample_options();
        let filter = CompiledFilter::new("type == 'put'").unwrap();

        let result = compute_leg_entries(
            &opts,
            &["A".into()], // A is held
            &filter,
            "optionroot",
            "ask",
            None,
            true,
            100,
            false,
        )
        .unwrap();

        // A is excluded, C is a call (filtered out), so B and D remain
        assert_eq!(result.height(), 2);
    }

    #[test]
    fn compute_qty() {
        let costs = Series::new("cost".into(), &[100.0, 200.0, 50.0]);
        let qty = compute_entry_qty(&costs, 1000.0).unwrap();
        let vals: Vec<f64> = qty.f64().unwrap().into_no_null_iter().collect();
        assert_eq!(vals, vec![10.0, 5.0, 20.0]);
    }
}
