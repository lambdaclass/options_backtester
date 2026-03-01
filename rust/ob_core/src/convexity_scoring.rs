/// Convexity ratio scoring: find cheapest tail protection per day.

pub struct DailyScore {
    pub date_ns: i64,
    pub convexity_ratio: f64,
    pub strike: f64,
    pub ask: f64,
    pub bid: f64,
    pub delta: f64,
    pub underlying_price: f64,
    pub implied_vol: f64,
    pub dte: i32,
    pub annual_cost: f64,
    pub tail_payoff: f64,
}

/// Find the put closest to target_delta within DTE range.
/// Returns the index within the provided slices, or None.
pub fn find_target_put(
    deltas: &[f64],
    dtes: &[i32],
    asks: &[f64],
    target_delta: f64,
    dte_min: i32,
    dte_max: i32,
) -> Option<usize> {
    let mut best_idx: Option<usize> = None;
    let mut best_delta_diff = f64::MAX;

    for i in 0..deltas.len() {
        if dtes[i] < dte_min || dtes[i] > dte_max {
            continue;
        }
        if asks[i] <= 0.0 || asks[i].is_nan() {
            continue;
        }
        if deltas[i].is_nan() {
            continue;
        }

        let delta_diff = (deltas[i] - target_delta).abs();
        if delta_diff < best_delta_diff {
            best_delta_diff = delta_diff;
            best_idx = Some(i);
        }
    }

    best_idx
}

/// Compute convexity ratio for a single put.
/// Returns (ratio, tail_payoff, annual_cost).
pub fn convexity_ratio(strike: f64, underlying: f64, ask: f64, tail_drop: f64) -> (f64, f64, f64) {
    let tail_price = underlying * (1.0 - tail_drop);
    let tail_payoff = (strike - tail_price).max(0.0) * 100.0;
    let annual_cost = ask * 100.0 * 12.0;
    let ratio = if annual_cost > 0.0 {
        tail_payoff / annual_cost
    } else {
        0.0
    };
    (ratio, tail_payoff, annual_cost)
}

/// Compute daily convexity scores from sorted puts data.
/// Input arrays must be sorted by date. Only put options should be passed.
pub fn compute_daily_scores(
    dates_ns: &[i64],
    strikes: &[f64],
    bids: &[f64],
    asks: &[f64],
    deltas: &[f64],
    underlying_prices: &[f64],
    dtes: &[i32],
    implied_vols: &[f64],
    target_delta: f64,
    dte_min: i32,
    dte_max: i32,
    tail_drop: f64,
) -> Vec<DailyScore> {
    let n = dates_ns.len();
    let mut results = Vec::new();

    if n == 0 {
        return results;
    }

    // Walk through date groups (consecutive rows with same date)
    let mut start = 0;
    while start < n {
        let current_date = dates_ns[start];
        let mut end = start + 1;
        while end < n && dates_ns[end] == current_date {
            end += 1;
        }

        // Find target put in this date's options
        if let Some(rel_idx) = find_target_put(
            &deltas[start..end],
            &dtes[start..end],
            &asks[start..end],
            target_delta,
            dte_min,
            dte_max,
        ) {
            let idx = start + rel_idx;
            let (ratio, tail_payoff, annual_cost) =
                convexity_ratio(strikes[idx], underlying_prices[idx], asks[idx], tail_drop);

            results.push(DailyScore {
                date_ns: current_date,
                convexity_ratio: ratio,
                strike: strikes[idx],
                ask: asks[idx],
                bid: bids[idx],
                delta: deltas[idx],
                underlying_price: underlying_prices[idx],
                implied_vol: implied_vols[idx],
                dte: dtes[idx],
                annual_cost,
                tail_payoff,
            });
        }

        start = end;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convexity_ratio_basic() {
        let (ratio, payoff, cost) = convexity_ratio(360.0, 400.0, 3.0, 0.20);
        assert!((payoff - 4000.0).abs() < 0.01);
        assert!((cost - 3600.0).abs() < 0.01);
        assert!((ratio - 1.111).abs() < 0.01);
    }

    #[test]
    fn test_convexity_ratio_otm_after_crash() {
        let (ratio, payoff, _) = convexity_ratio(300.0, 400.0, 3.0, 0.20);
        assert_eq!(payoff, 0.0);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_find_target_put() {
        let deltas = vec![-0.05, -0.10, -0.15, -0.25, -0.50];
        let dtes = vec![30, 30, 30, 30, 30];
        let asks = vec![1.0, 2.0, 3.0, 5.0, 10.0];

        let idx = find_target_put(&deltas, &dtes, &asks, -0.10, 20, 45);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_find_target_put_dte_filter() {
        let deltas = vec![-0.10, -0.10, -0.10];
        let dtes = vec![10, 30, 60];
        let asks = vec![1.0, 2.0, 3.0];

        let idx = find_target_put(&deltas, &dtes, &asks, -0.10, 20, 45);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_find_target_put_skips_zero_ask() {
        let deltas = vec![-0.10, -0.11];
        let dtes = vec![30, 30];
        let asks = vec![0.0, 2.0];

        let idx = find_target_put(&deltas, &dtes, &asks, -0.10, 20, 45);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_compute_daily_scores() {
        let dates_ns = vec![100, 100, 100, 200, 200, 200];
        let strikes = vec![360.0, 370.0, 380.0, 360.0, 370.0, 380.0];
        let bids = vec![2.5, 3.5, 5.0, 2.0, 3.0, 4.5];
        let asks = vec![3.0, 4.0, 5.5, 2.5, 3.5, 5.0];
        let deltas = vec![-0.08, -0.12, -0.18, -0.09, -0.11, -0.17];
        let underlying = vec![400.0; 6];
        let dtes = vec![30, 30, 30, 30, 30, 30];
        let ivs = vec![0.20, 0.22, 0.25, 0.19, 0.21, 0.24];

        let scores = compute_daily_scores(
            &dates_ns, &strikes, &bids, &asks, &deltas, &underlying, &dtes, &ivs, -0.10, 20, 45,
            0.20,
        );

        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].date_ns, 100);
        assert_eq!(scores[1].date_ns, 200);
    }
}
