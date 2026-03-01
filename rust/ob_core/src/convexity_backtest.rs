/// Backtest engine: monthly rebalance loop for tail hedge overlay.
///
/// Model: 100% invested in equity (SPY). Each month, sell a fixed budget
/// worth of equity to buy ~10-delta puts. Put proceeds are reinvested
/// into equity at settlement. Budget is fixed at initial_capital * budget_pct
/// (not scaled with portfolio growth) to avoid unrealistic compounding.

use chrono::{DateTime, Datelike};

use crate::convexity_scoring;

struct Position {
    strike: f64,
    expiration_ns: i64,
    entry_ask: f64,
    contracts: i32,
}

pub struct MonthRecord {
    pub date_ns: i64,
    pub shares: f64,
    pub stock_price: f64,
    pub equity_value: f64,
    pub put_cost: f64,
    pub put_exit_value: f64,
    pub put_pnl: f64,
    pub portfolio_value: f64,
    pub convexity_ratio: f64,
    pub strike: f64,
    pub contracts: i32,
}

pub struct BacktestResult {
    pub records: Vec<MonthRecord>,
    pub daily_dates_ns: Vec<i64>,
    pub daily_balances: Vec<f64>,
}

fn ns_to_year_month(ns: i64) -> (i32, u32) {
    let secs = ns / 1_000_000_000;
    let dt = DateTime::from_timestamp(secs, 0).expect("valid timestamp");
    (dt.year(), dt.month())
}

/// Extract monthly rebalance dates (first trading day of each month).
fn monthly_rebalance_dates(stock_dates_ns: &[i64]) -> Vec<i64> {
    let mut dates = Vec::new();
    if stock_dates_ns.is_empty() {
        return dates;
    }

    let mut prev_ym = ns_to_year_month(stock_dates_ns[0]);
    dates.push(stock_dates_ns[0]);

    for &d in &stock_dates_ns[1..] {
        let ym = ns_to_year_month(d);
        if ym != prev_ym {
            dates.push(d);
            prev_ym = ym;
        }
    }

    dates
}

/// Binary search for first index where arr[i] >= target.
fn lower_bound(arr: &[i64], target: i64) -> usize {
    arr.partition_point(|&x| x < target)
}

/// Find the index range [start, end) for a specific date in sorted data.
fn find_date_range(dates_ns: &[i64], target: i64) -> (usize, usize) {
    let start = lower_bound(dates_ns, target);
    if start >= dates_ns.len() || dates_ns[start] != target {
        return (start, start);
    }
    let mut end = start + 1;
    while end < dates_ns.len() && dates_ns[end] == target {
        end += 1;
    }
    (start, end)
}

/// Find stock price on or before a given date.
fn stock_price_on(stock_dates_ns: &[i64], stock_prices: &[f64], target_ns: i64) -> Option<f64> {
    let idx = lower_bound(stock_dates_ns, target_ns);
    if idx < stock_dates_ns.len() && stock_dates_ns[idx] == target_ns {
        Some(stock_prices[idx])
    } else if idx > 0 {
        Some(stock_prices[idx - 1])
    } else {
        None
    }
}

/// Close a position: find exit value from options data or use intrinsic.
fn close_position(
    pos: &Position,
    rebal_date_ns: i64,
    put_dates_ns: &[i64],
    put_expirations_ns: &[i64],
    put_strikes: &[f64],
    put_bids: &[f64],
    stock_dates_ns: &[i64],
    stock_prices: &[f64],
    current_stock_price: f64,
) -> f64 {
    if pos.expiration_ns <= rebal_date_ns {
        let exp_price = stock_price_on(stock_dates_ns, stock_prices, pos.expiration_ns)
            .unwrap_or(current_stock_price);
        let intrinsic = (pos.strike - exp_price).max(0.0);
        intrinsic * 100.0 * pos.contracts as f64
    } else {
        let (start, end) = find_date_range(put_dates_ns, rebal_date_ns);
        for j in start..end {
            if (put_strikes[j] - pos.strike).abs() < 0.001
                && put_expirations_ns[j] == pos.expiration_ns
            {
                return put_bids[j] * 100.0 * pos.contracts as f64;
            }
        }
        let intrinsic = (pos.strike - current_stock_price).max(0.0);
        intrinsic * 100.0 * pos.contracts as f64
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_backtest(
    put_dates_ns: &[i64],
    put_expirations_ns: &[i64],
    put_strikes: &[f64],
    put_bids: &[f64],
    put_asks: &[f64],
    put_deltas: &[f64],
    put_underlying: &[f64],
    put_dtes: &[i32],
    _put_ivs: &[f64],
    stock_dates_ns: &[i64],
    stock_prices: &[f64],
    initial_capital: f64,
    budget_pct: f64,
    target_delta: f64,
    dte_min: i32,
    dte_max: i32,
    tail_drop: f64,
) -> BacktestResult {
    let rebalance_dates = monthly_rebalance_dates(stock_dates_ns);

    let mut records = Vec::with_capacity(rebalance_dates.len());
    let mut daily_dates: Vec<i64> = Vec::with_capacity(stock_dates_ns.len());
    let mut daily_balances: Vec<f64> = Vec::with_capacity(stock_dates_ns.len());

    if rebalance_dates.is_empty() || stock_dates_ns.is_empty() {
        return BacktestResult {
            records,
            daily_dates_ns: daily_dates,
            daily_balances,
        };
    }

    let first_price = stock_price_on(stock_dates_ns, stock_prices, rebalance_dates[0])
        .unwrap_or(stock_prices[0]);
    let mut shares = initial_capital / first_price;
    let mut position: Option<Position> = None;
    let fixed_budget = initial_capital * budget_pct;

    for (i, &rebal_date) in rebalance_dates.iter().enumerate() {
        let stock_price =
            stock_price_on(stock_dates_ns, stock_prices, rebal_date).unwrap_or(first_price);

        // 1. Close existing position — reinvest proceeds into equity
        let (put_exit_value, prev_put_cost) = if let Some(ref pos) = position {
            let cost = pos.entry_ask * 100.0 * pos.contracts as f64;
            let exit_val = close_position(
                pos,
                rebal_date,
                put_dates_ns,
                put_expirations_ns,
                put_strikes,
                put_bids,
                stock_dates_ns,
                stock_prices,
                stock_price,
            );
            if stock_price > 0.0 {
                shares += exit_val / stock_price;
            }
            (exit_val, cost)
        } else {
            (0.0, 0.0)
        };
        let put_pnl = put_exit_value - prev_put_cost;

        // 2. Fixed budget — sell equity worth fixed_budget to fund puts
        let budget = fixed_budget;
        if stock_price > 0.0 {
            shares -= budget / stock_price;
        }

        // 3. Open new position
        let (opt_start, opt_end) = find_date_range(put_dates_ns, rebal_date);

        let mut new_cost = 0.0;
        let mut new_contracts = 0i32;
        let mut new_strike = 0.0;
        let mut new_ratio = 0.0;

        if opt_start < opt_end {
            let slice_deltas = &put_deltas[opt_start..opt_end];
            let slice_dtes = &put_dtes[opt_start..opt_end];
            let slice_asks = &put_asks[opt_start..opt_end];

            if let Some(rel_idx) = convexity_scoring::find_target_put(
                slice_deltas,
                slice_dtes,
                slice_asks,
                target_delta,
                dte_min,
                dte_max,
            ) {
                let idx = opt_start + rel_idx;
                let ask = put_asks[idx];
                let strike = put_strikes[idx];
                let underlying = put_underlying[idx];

                if ask > 0.0 {
                    let contracts = (budget / (ask * 100.0)) as i32;
                    if contracts > 0 {
                        let cost = ask * 100.0 * contracts as f64;
                        new_cost = cost;
                        new_contracts = contracts;
                        new_strike = strike;

                        let (ratio, _, _) =
                            convexity_scoring::convexity_ratio(strike, underlying, ask, tail_drop);
                        new_ratio = ratio;

                        position = Some(Position {
                            strike,
                            expiration_ns: put_expirations_ns[idx],
                            entry_ask: ask,
                            contracts,
                        });

                        // Reinvest leftover
                        let leftover = budget - cost;
                        if stock_price > 0.0 {
                            shares += leftover / stock_price;
                        }
                    } else {
                        position = None;
                        if stock_price > 0.0 {
                            shares += budget / stock_price;
                        }
                    }
                } else {
                    position = None;
                    if stock_price > 0.0 {
                        shares += budget / stock_price;
                    }
                }
            } else {
                position = None;
                if stock_price > 0.0 {
                    shares += budget / stock_price;
                }
            }
        } else {
            position = None;
            if stock_price > 0.0 {
                shares += budget / stock_price;
            }
        }

        let final_value = shares * stock_price;

        records.push(MonthRecord {
            date_ns: rebal_date,
            shares,
            stock_price,
            equity_value: final_value,
            put_cost: new_cost,
            put_exit_value,
            put_pnl,
            portfolio_value: final_value,
            convexity_ratio: new_ratio,
            strike: new_strike,
            contracts: new_contracts,
        });

        // 4. Record daily balances until next rebalance
        let stock_idx = lower_bound(stock_dates_ns, rebal_date);
        let next_rebal = if i + 1 < rebalance_dates.len() {
            rebalance_dates[i + 1]
        } else {
            i64::MAX
        };

        for si in stock_idx..stock_dates_ns.len() {
            if stock_dates_ns[si] >= next_rebal {
                break;
            }
            daily_dates.push(stock_dates_ns[si]);
            daily_balances.push(shares * stock_prices[si]);
        }
    }

    BacktestResult {
        records,
        daily_dates_ns: daily_dates,
        daily_balances,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ts(year: i32, month: u32, day: u32) -> i64 {
        use chrono::NaiveDate;
        let dt = NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        dt.and_utc().timestamp_nanos_opt().unwrap()
    }

    #[test]
    fn test_monthly_rebalance_dates() {
        let dates = vec![
            make_ts(2020, 1, 2),
            make_ts(2020, 1, 3),
            make_ts(2020, 1, 6),
            make_ts(2020, 2, 3),
            make_ts(2020, 2, 4),
            make_ts(2020, 3, 2),
        ];
        let rebal = monthly_rebalance_dates(&dates);
        assert_eq!(rebal.len(), 3);
        assert_eq!(rebal[0], dates[0]);
        assert_eq!(rebal[1], dates[3]);
        assert_eq!(rebal[2], dates[5]);
    }

    #[test]
    fn test_stock_price_on_exact() {
        let dates = vec![100, 200, 300];
        let prices = vec![10.0, 20.0, 30.0];
        assert_eq!(stock_price_on(&dates, &prices, 200), Some(20.0));
    }

    #[test]
    fn test_stock_price_on_before() {
        let dates = vec![100, 200, 300];
        let prices = vec![10.0, 20.0, 30.0];
        assert_eq!(stock_price_on(&dates, &prices, 250), Some(20.0));
    }

    #[test]
    fn test_run_backtest_no_options() {
        let stock_dates = vec![make_ts(2020, 1, 2), make_ts(2020, 2, 3)];
        let stock_prices = vec![100.0, 105.0];

        let result = run_backtest(
            &[], &[], &[], &[], &[], &[], &[], &[], &[],
            &stock_dates, &stock_prices,
            100_000.0, 0.005, -0.10, 14, 60, 0.20,
        );

        assert_eq!(result.records.len(), 2);
        assert!(result.records[0].contracts == 0);
    }
}
