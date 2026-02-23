/// Performance statistics computation.
///
/// Sharpe, Sortino, Calmar ratios, max drawdown, profit factor — all vectorized.

use crate::types::Stats;

const TRADING_DAYS_PER_YEAR: f64 = 252.0;

/// Compute all stats from daily returns and trade PnLs.
pub fn compute_stats(
    daily_returns: &[f64],
    trade_pnls: &[f64],
    risk_free_rate: f64,
) -> Stats {
    let n = daily_returns.len();
    if n == 0 {
        return Stats::default();
    }

    let total_return = daily_returns
        .iter()
        .fold(1.0, |acc, &r| acc * (1.0 + r))
        - 1.0;

    let years = n as f64 / TRADING_DAYS_PER_YEAR;
    let annualized_return = if years > 0.0 {
        (1.0 + total_return).powf(1.0 / years) - 1.0
    } else {
        0.0
    };

    let daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR;
    let excess_returns: Vec<f64> = daily_returns.iter().map(|&r| r - daily_rf).collect();

    let mean_excess = mean(&excess_returns);
    let excess_std = std_dev(&excess_returns);
    let sharpe_ratio = if excess_std > 0.0 {
        mean_excess / excess_std * TRADING_DAYS_PER_YEAR.sqrt()
    } else {
        0.0
    };

    let downside: Vec<f64> = excess_returns
        .iter()
        .filter(|&&r| r < 0.0)
        .copied()
        .collect();
    let downside_std = std_dev(&downside);
    let sortino_ratio = if downside_std > 0.0 {
        mean_excess / downside_std * TRADING_DAYS_PER_YEAR.sqrt()
    } else {
        0.0
    };

    let (max_drawdown, max_drawdown_duration) = compute_drawdown(daily_returns);
    let calmar_ratio = if max_drawdown > 0.0 {
        annualized_return / max_drawdown
    } else {
        0.0
    };

    let (gross_profit, gross_loss, wins, total_trades) = trade_stats(trade_pnls);
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        0.0
    };
    let win_rate = if total_trades > 0 {
        wins as f64 / total_trades as f64
    } else {
        0.0
    };

    Stats {
        total_return,
        annualized_return,
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        max_drawdown,
        max_drawdown_duration,
        profit_factor,
        win_rate,
        total_trades,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn compute_drawdown(daily_returns: &[f64]) -> (f64, u32) {
    let mut peak = 1.0;
    let mut equity = 1.0;
    let mut max_dd = 0.0;
    let mut max_dd_dur: u32 = 0;
    let mut current_dur: u32 = 0;

    for &r in daily_returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
            current_dur = 0;
        } else {
            current_dur += 1;
            max_dd_dur = max_dd_dur.max(current_dur);
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    (max_dd, max_dd_dur)
}

fn trade_stats(pnls: &[f64]) -> (f64, f64, u32, u32) {
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut wins: u32 = 0;

    for &pnl in pnls {
        if pnl > 0.0 {
            gross_profit += pnl;
            wins += 1;
        } else {
            gross_loss += pnl.abs();
        }
    }

    (gross_profit, gross_loss, wins, pnls.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_empty() {
        let s = compute_stats(&[], &[], 0.0);
        assert_eq!(s.total_return, 0.0);
    }

    #[test]
    fn stats_simple_returns() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let s = compute_stats(&returns, &[], 0.0);
        assert!(s.total_return > 0.0);
        assert!(s.sharpe_ratio != 0.0);
    }

    #[test]
    fn drawdown_calculation() {
        // 10% gain then 20% loss from peak
        let returns = vec![0.10, -0.18182]; // 1.0 -> 1.1 -> 0.9
        let (dd, _dur) = compute_drawdown(&returns);
        assert!((dd - 0.18182).abs() < 0.01);
    }

    #[test]
    fn profit_factor_calculation() {
        let pnls = vec![100.0, -50.0, 200.0, -30.0];
        let s = compute_stats(&[0.01; 4], &pnls, 0.0);
        assert!((s.profit_factor - 300.0 / 80.0).abs() < 0.01);
        assert_eq!(s.total_trades, 4);
        assert_eq!(s.win_rate, 0.5);
    }
}
