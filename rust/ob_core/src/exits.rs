//! Exit mask computation in Rust.
//!
//! Mirrors Python's _execute_option_exits:
//! 1. Compute current option quotes for each leg
//! 2. Apply exit filters to get filter masks
//! 3. Apply threshold exits (profit/loss targets)
//! 4. Combine masks with OR

use polars::prelude::*;

/// Compute exit mask from profit/loss thresholds.
///
/// exit if: current_cost <= entry_cost * (1 - loss_pct)  [loss]
///      or: current_cost >= entry_cost * (1 + profit_pct)  [profit]
pub fn threshold_exit_mask(
    entry_costs: &Series,
    current_costs: &Series,
    profit_pct: Option<f64>,
    loss_pct: Option<f64>,
) -> PolarsResult<BooleanChunked> {
    let entry = entry_costs.f64()?;
    let current = current_costs.f64()?;

    let mask: BooleanChunked = entry
        .into_iter()
        .zip(current.into_iter())
        .map(|(e, c)| {
            match (e, c) {
                (Some(entry_val), Some(curr_val)) => {
                    let mut should_exit = false;
                    if let Some(p) = profit_pct {
                        if entry_val != 0.0 {
                            let pnl_pct = (curr_val - entry_val) / entry_val.abs();
                            if pnl_pct >= p {
                                should_exit = true;
                            }
                        }
                    }
                    if let Some(l) = loss_pct {
                        if entry_val != 0.0 {
                            let pnl_pct = (curr_val - entry_val) / entry_val.abs();
                            if pnl_pct <= -l {
                                should_exit = true;
                            }
                        }
                    }
                    Some(should_exit)
                }
                _ => Some(false),
            }
        })
        .collect();

    Ok(mask)
}

/// Combine multiple boolean masks with OR.
pub fn combine_masks_or(masks: &[BooleanChunked]) -> BooleanChunked {
    if masks.is_empty() {
        return BooleanChunked::new("mask".into(), &[] as &[bool]);
    }
    let mut result = masks[0].clone();
    for mask in &masks[1..] {
        result = result | mask.clone();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_profit_exit() {
        let entry = Series::new("entry".into(), &[100.0, 100.0, 100.0]);
        let current = Series::new("current".into(), &[160.0, 110.0, 80.0]);

        let mask = threshold_exit_mask(&entry, &current, Some(0.50), None).unwrap();
        let vals: Vec<bool> = mask.into_no_null_iter().collect();
        assert_eq!(vals, vec![true, false, false]); // 60% profit > 50%
    }

    #[test]
    fn threshold_loss_exit() {
        let entry = Series::new("entry".into(), &[100.0, 100.0, 100.0]);
        let current = Series::new("current".into(), &[160.0, 110.0, 70.0]);

        let mask = threshold_exit_mask(&entry, &current, None, Some(0.20)).unwrap();
        let vals: Vec<bool> = mask.into_no_null_iter().collect();
        assert_eq!(vals, vec![false, false, true]); // 30% loss > 20%
    }

    #[test]
    fn combine_masks() {
        let a = BooleanChunked::new("a".into(), &[true, false, false]);
        let b = BooleanChunked::new("b".into(), &[false, false, true]);

        let result = combine_masks_or(&[a, b]);
        let vals: Vec<bool> = result.into_no_null_iter().collect();
        assert_eq!(vals, vec![true, false, true]);
    }
}
