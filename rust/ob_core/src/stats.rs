//! Performance statistics computation.
//!
//! Comprehensive stats matching Python's BacktestStats: return metrics,
//! drawdown analysis, period stats, lookback returns, trade stats,
//! portfolio metrics (turnover, Herfindahl).

const TRADING_DAYS_PER_YEAR: f64 = 252.0;
const MONTHS_PER_YEAR: f64 = 12.0;

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Stats for a specific return frequency (daily, monthly, yearly).
#[derive(Debug, Clone, Default)]
pub struct PeriodStats {
    pub mean: f64,
    pub vol: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub skew: f64,
    pub kurtosis: f64,
    pub best: f64,
    pub worst: f64,
}

/// Trailing-period returns as of the last date.
#[derive(Debug, Clone, Default)]
pub struct LookbackReturns {
    pub mtd: Option<f64>,
    pub three_month: Option<f64>,
    pub six_month: Option<f64>,
    pub ytd: Option<f64>,
    pub one_year: Option<f64>,
    pub three_year: Option<f64>,
    pub five_year: Option<f64>,
    pub ten_year: Option<f64>,
}

/// Comprehensive backtest statistics.
#[derive(Debug, Clone, Default)]
pub struct FullStats {
    // Trade stats
    pub total_trades: u32,
    pub wins: u32,
    pub losses: u32,
    pub win_pct: f64,
    pub profit_factor: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub avg_trade: f64,

    // Return stats
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,

    // Risk stats
    pub max_drawdown: f64,
    pub max_drawdown_duration: u32,
    pub avg_drawdown: f64,
    pub avg_drawdown_duration: u32,
    pub volatility: f64,
    pub tail_ratio: f64,

    // Period stats
    pub daily: PeriodStats,
    pub monthly: PeriodStats,
    pub yearly: PeriodStats,

    // Lookback
    pub lookback: LookbackReturns,

    // Portfolio metrics
    pub turnover: f64,
    pub herfindahl: f64,
}

// ---------------------------------------------------------------------------
// Legacy Stats (kept for backward compat with existing callers)
// ---------------------------------------------------------------------------

/// Legacy stats struct used by run_backtest_py / parallel_sweep.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: u32,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub total_trades: u32,
}

// ---------------------------------------------------------------------------
// Main entry points
// ---------------------------------------------------------------------------

/// Compute legacy stats (backward compat).
pub fn compute_stats(
    daily_returns: &[f64],
    trade_pnls: &[f64],
    risk_free_rate: f64,
) -> Stats {
    let n = daily_returns.len();
    if n == 0 {
        return Stats::default();
    }

    let total_return = cum_return(daily_returns);
    let years = n as f64 / TRADING_DAYS_PER_YEAR;
    let annualized_return = annualize(total_return, years);

    let sharpe_ratio = sharpe(daily_returns, risk_free_rate, TRADING_DAYS_PER_YEAR);
    let sortino_ratio = sortino(daily_returns, risk_free_rate, TRADING_DAYS_PER_YEAR);

    let dd = compute_drawdown_full(daily_returns);
    let calmar_ratio = if dd.max_drawdown > 0.0 {
        annualized_return / dd.max_drawdown
    } else {
        0.0
    };

    let ts = compute_trade_stats(trade_pnls);

    Stats {
        total_return,
        annualized_return,
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        max_drawdown: dd.max_drawdown,
        max_drawdown_duration: dd.max_drawdown_duration,
        profit_factor: ts.profit_factor,
        win_rate: ts.win_pct / 100.0, // legacy uses 0-1 scale
        total_trades: ts.total_trades,
    }
}

/// Compute comprehensive stats from total_capital series + optional trade PnLs.
///
/// `total_capital`: daily total capital values (one per trading day).
/// `timestamps_ns`: nanosecond timestamps for each capital value (for monthly/yearly resampling).
/// `trade_pnls`: per-trade profit/loss values.
/// `stock_weights`: flattened [n_days × n_stocks] matrix of portfolio weights (row-major).
/// `n_stocks`: number of stock columns.
/// `risk_free_rate`: annualized risk-free rate.
pub fn compute_full_stats(
    total_capital: &[f64],
    timestamps_ns: &[i64],
    trade_pnls: &[f64],
    stock_weights: &[f64],
    n_stocks: usize,
    risk_free_rate: f64,
) -> FullStats {
    let mut fs = FullStats::default();

    if total_capital.len() < 2 {
        return fs;
    }

    // Daily returns from capital series
    let daily_returns: Vec<f64> = total_capital
        .windows(2)
        .map(|w| if w[0] != 0.0 { w[1] / w[0] - 1.0 } else { 0.0 })
        .collect();

    let n = daily_returns.len();
    if n == 0 {
        return fs;
    }

    // -- Return metrics --
    fs.total_return = total_capital.last().unwrap() / total_capital[0] - 1.0;
    let years = n as f64 / TRADING_DAYS_PER_YEAR;
    fs.annualized_return = annualize(fs.total_return, years);
    fs.volatility = std_dev(&daily_returns) * TRADING_DAYS_PER_YEAR.sqrt();
    fs.sharpe_ratio = sharpe(&daily_returns, risk_free_rate, TRADING_DAYS_PER_YEAR);
    fs.sortino_ratio = sortino(&daily_returns, risk_free_rate, TRADING_DAYS_PER_YEAR);

    // -- Drawdown --
    let dd = compute_drawdown_full(&daily_returns);
    fs.max_drawdown = dd.max_drawdown;
    fs.max_drawdown_duration = dd.max_drawdown_duration;
    fs.avg_drawdown = dd.avg_drawdown;
    fs.avg_drawdown_duration = dd.avg_drawdown_duration;

    // Calmar
    if fs.max_drawdown > 0.0 {
        fs.calmar_ratio = fs.annualized_return / fs.max_drawdown;
    }

    // Tail ratio
    if n > 20 {
        let p95 = percentile(&daily_returns, 95.0);
        let p5 = percentile(&daily_returns, 5.0).abs();
        if p5 > 0.0 {
            fs.tail_ratio = p95 / p5;
        }
    }

    // -- Daily period stats --
    fs.daily = compute_period_stats(&daily_returns, risk_free_rate, TRADING_DAYS_PER_YEAR);

    // -- Monthly period stats --
    let monthly_returns = resample_returns(total_capital, timestamps_ns, ResampleFreq::Monthly);
    if !monthly_returns.is_empty() {
        fs.monthly = compute_period_stats(&monthly_returns, risk_free_rate, MONTHS_PER_YEAR);
    }

    // -- Yearly period stats --
    let yearly_returns = resample_returns(total_capital, timestamps_ns, ResampleFreq::Yearly);
    if !yearly_returns.is_empty() {
        fs.yearly = compute_period_stats(&yearly_returns, risk_free_rate, 1.0);
    }

    // -- Lookback returns --
    fs.lookback = compute_lookback(total_capital, timestamps_ns);

    // -- Turnover --
    fs.turnover = compute_turnover(stock_weights, n_stocks);

    // -- Herfindahl --
    fs.herfindahl = compute_herfindahl(stock_weights, n_stocks);

    // -- Trade stats --
    let ts = compute_trade_stats(trade_pnls);
    fs.total_trades = ts.total_trades;
    fs.wins = ts.wins;
    fs.losses = ts.losses;
    fs.win_pct = ts.win_pct;
    fs.profit_factor = ts.profit_factor;
    fs.largest_win = ts.largest_win;
    fs.largest_loss = ts.largest_loss;
    fs.avg_win = ts.avg_win;
    fs.avg_loss = ts.avg_loss;
    fs.avg_trade = ts.avg_trade;

    fs
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn cum_return(returns: &[f64]) -> f64 {
    returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0
}

fn annualize(total_return: f64, years: f64) -> f64 {
    if years > 0.0 {
        (1.0 + total_return).powf(1.0 / years) - 1.0
    } else {
        0.0
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
    let variance =
        values.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn skewness(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return 0.0;
    }
    let m = mean(values);
    let s = std_dev(values);
    if s == 0.0 {
        return 0.0;
    }
    let nf = n as f64;
    let m3: f64 = values.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / nf;
    // Adjusted Fisher-Pearson (matches pandas default)
    let adj = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    adj * m3
}

fn kurtosis_excess(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return 0.0;
    }
    let m = mean(values);
    let s = std_dev(values);
    if s == 0.0 {
        return 0.0;
    }
    let nf = n as f64;
    let m4: f64 = values.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / nf;
    // Excess kurtosis with bias correction (matches pandas default)
    let raw = m4 - 3.0;
    let adj = (nf - 1.0) / ((nf - 2.0) * (nf - 3.0)) * ((nf + 1.0) * raw + 6.0);
    adj
}

fn sharpe(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let rf_per_period = (1.0 + risk_free_rate).powf(1.0 / periods_per_year) - 1.0;
    let excess: Vec<f64> = returns.iter().map(|&r| r - rf_per_period).collect();
    let s = std_dev(&excess);
    if s == 0.0 {
        return 0.0;
    }
    mean(&excess) / s * periods_per_year.sqrt()
}

fn sortino(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let rf_per_period = (1.0 + risk_free_rate).powf(1.0 / periods_per_year) - 1.0;
    let excess: Vec<f64> = returns.iter().map(|&r| r - rf_per_period).collect();
    let downside: Vec<f64> = excess.iter().filter(|&&r| r < 0.0).copied().collect();
    if downside.is_empty() {
        return 0.0;
    }
    let s = std_dev(&downside);
    if s == 0.0 {
        return 0.0;
    }
    mean(&excess) / s * periods_per_year.sqrt()
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi || hi >= sorted.len() {
        sorted[lo.min(sorted.len() - 1)]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// -- Drawdown --

struct DrawdownResult {
    max_drawdown: f64,
    max_drawdown_duration: u32,
    avg_drawdown: f64,
    avg_drawdown_duration: u32,
}

fn compute_drawdown_full(daily_returns: &[f64]) -> DrawdownResult {
    let mut peak = 1.0_f64;
    let mut equity = 1.0_f64;
    let mut max_dd = 0.0_f64;
    let mut max_dd_dur: u32 = 0;
    let mut current_dur: u32 = 0;

    // Track drawdown episodes for avg computation
    let mut episode_depths: Vec<f64> = Vec::new();
    let mut episode_durations: Vec<u32> = Vec::new();
    let mut current_min_dd = 0.0_f64; // deepest dd in current episode

    for &r in daily_returns {
        equity *= 1.0 + r;
        if equity > peak {
            // End of drawdown episode (if we were in one)
            if current_dur > 0 {
                episode_depths.push(current_min_dd);
                episode_durations.push(current_dur);
            }
            peak = equity;
            current_dur = 0;
            current_min_dd = 0.0;
        } else {
            current_dur += 1;
            max_dd_dur = max_dd_dur.max(current_dur);
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
        if dd > current_min_dd {
            current_min_dd = dd;
        }
    }
    // Close last episode if still in drawdown
    if current_dur > 0 {
        episode_depths.push(current_min_dd);
        episode_durations.push(current_dur);
    }

    let avg_drawdown = if episode_depths.is_empty() {
        0.0
    } else {
        mean(&episode_depths)
    };

    let avg_drawdown_duration = if episode_durations.is_empty() {
        0
    } else {
        let dur_f: Vec<f64> = episode_durations.iter().map(|&d| d as f64).collect();
        mean(&dur_f) as u32
    };

    DrawdownResult {
        max_drawdown: max_dd,
        max_drawdown_duration: max_dd_dur,
        avg_drawdown,
        avg_drawdown_duration,
    }
}

// -- Period stats --

fn compute_period_stats(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> PeriodStats {
    if returns.is_empty() {
        return PeriodStats::default();
    }
    PeriodStats {
        mean: mean(returns),
        vol: std_dev(returns),
        sharpe: sharpe(returns, risk_free_rate, periods_per_year),
        sortino: sortino(returns, risk_free_rate, periods_per_year),
        skew: skewness(returns),
        kurtosis: kurtosis_excess(returns),
        best: returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        worst: returns.iter().cloned().fold(f64::INFINITY, f64::min),
    }
}

// -- Resampling --

#[derive(Clone, Copy)]
enum ResampleFreq {
    Monthly,
    Yearly,
}

/// Resample total_capital to end-of-period values, then compute returns.
fn resample_returns(
    total_capital: &[f64],
    timestamps_ns: &[i64],
    freq: ResampleFreq,
) -> Vec<f64> {
    if total_capital.len() < 2 || timestamps_ns.len() != total_capital.len() {
        return Vec::new();
    }

    // Group by period key, take last value in each period
    let mut period_vals: Vec<f64> = Vec::new();
    let mut last_key: Option<(i32, u32)> = None;

    for (i, &ts_ns) in timestamps_ns.iter().enumerate() {
        let key = period_key(ts_ns, freq);
        match last_key {
            Some(prev) if prev != key => {
                // Previous period ended at i-1
                period_vals.push(total_capital[i - 1]);
                last_key = Some(key);
            }
            None => {
                last_key = Some(key);
            }
            _ => {}
        }
    }
    // Push the last period value
    if let Some(_) = last_key {
        period_vals.push(*total_capital.last().unwrap());
    }

    // Compute returns from period values
    if period_vals.len() < 2 {
        return Vec::new();
    }
    period_vals
        .windows(2)
        .map(|w| if w[0] != 0.0 { w[1] / w[0] - 1.0 } else { 0.0 })
        .collect()
}

/// Convert nanosecond timestamp to (year, period) key.
fn period_key(ts_ns: i64, freq: ResampleFreq) -> (i32, u32) {
    // Convert nanoseconds since epoch to days
    let days_since_epoch = (ts_ns / 86_400_000_000_000) as i32;
    // Simple calendar calculation from days since 1970-01-01
    let (year, month, _day) = days_to_ymd(days_since_epoch);
    match freq {
        ResampleFreq::Monthly => (year, month),
        ResampleFreq::Yearly => (year, 0),
    }
}

/// Convert days since epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: i32) -> (i32, u32, u32) {
    // Algorithm from Howard Hinnant's date library (public domain)
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

// -- Lookback returns --

fn compute_lookback(total_capital: &[f64], timestamps_ns: &[i64]) -> LookbackReturns {
    let mut lb = LookbackReturns::default();
    if total_capital.len() < 2 || timestamps_ns.len() != total_capital.len() {
        return lb;
    }

    let end_val = *total_capital.last().unwrap();
    let end_ts = *timestamps_ns.last().unwrap();
    let (end_year, end_month, _end_day) = days_to_ymd((end_ts / 86_400_000_000_000) as i32);

    // Helper: find return since the first data point on or after target_ns
    let return_since = |target_ns: i64| -> Option<f64> {
        match timestamps_ns.iter().position(|&ts| ts >= target_ns) {
            Some(idx) => {
                let start_val = total_capital[idx];
                if start_val == 0.0 {
                    None
                } else {
                    Some(end_val / start_val - 1.0)
                }
            }
            None => None,
        }
    };

    // MTD: start of current month
    lb.mtd = return_since(ymd_to_ns(end_year, end_month, 1));

    // YTD: start of current year
    lb.ytd = return_since(ymd_to_ns(end_year, 1, 1));

    // Fixed offsets (in months)
    let offsets: [(fn(&mut LookbackReturns, Option<f64>), u32); 6] = [
        (|lb, v| lb.three_month = v, 3),
        (|lb, v| lb.six_month = v, 6),
        (|lb, v| lb.one_year = v, 12),
        (|lb, v| lb.three_year = v, 36),
        (|lb, v| lb.five_year = v, 60),
        (|lb, v| lb.ten_year = v, 120),
    ];

    for (setter, months) in offsets {
        let target_ns = subtract_months_ns(end_ts, months);
        setter(&mut lb, return_since(target_ns));
    }

    lb
}

fn ymd_to_ns(year: i32, month: u32, day: u32) -> i64 {
    // Inverse of days_to_ymd: compute days since epoch
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 { month + 9 } else { month - 3 } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86_400_000_000_000
}

fn subtract_months_ns(ts_ns: i64, months: u32) -> i64 {
    let (year, month, day) = days_to_ymd((ts_ns / 86_400_000_000_000) as i32);
    let total_months = year * 12 + month as i32 - months as i32;
    let new_year = if total_months > 0 {
        (total_months - 1) / 12
    } else {
        (total_months - 12) / 12
    };
    let new_month = total_months - new_year * 12;
    ymd_to_ns(new_year, new_month as u32, day.min(28)) // clamp day to avoid invalid dates
}

// -- Portfolio metrics --

fn compute_turnover(stock_weights: &[f64], n_stocks: usize) -> f64 {
    if n_stocks == 0 || stock_weights.is_empty() {
        return 0.0;
    }
    let n_days = stock_weights.len() / n_stocks;
    if n_days < 2 {
        return 0.0;
    }

    let mut total_change = 0.0;
    for day in 1..n_days {
        let mut day_change = 0.0;
        for s in 0..n_stocks {
            let prev = stock_weights[(day - 1) * n_stocks + s];
            let curr = stock_weights[day * n_stocks + s];
            day_change += (curr - prev).abs();
        }
        total_change += day_change;
    }

    total_change / (n_days - 1) as f64 / 2.0
}

fn compute_herfindahl(stock_weights: &[f64], n_stocks: usize) -> f64 {
    if n_stocks == 0 || stock_weights.is_empty() {
        return 0.0;
    }
    let n_days = stock_weights.len() / n_stocks;
    if n_days == 0 {
        return 0.0;
    }

    let mut total_hhi = 0.0;
    for day in 0..n_days {
        let mut hhi = 0.0;
        for s in 0..n_stocks {
            let w = stock_weights[day * n_stocks + s];
            hhi += w * w;
        }
        total_hhi += hhi;
    }

    total_hhi / n_days as f64
}

// -- Trade stats --

struct TradeStatsResult {
    total_trades: u32,
    wins: u32,
    losses: u32,
    win_pct: f64,
    profit_factor: f64,
    largest_win: f64,
    largest_loss: f64,
    avg_win: f64,
    avg_loss: f64,
    avg_trade: f64,
}

fn compute_trade_stats(pnls: &[f64]) -> TradeStatsResult {
    if pnls.is_empty() {
        return TradeStatsResult {
            total_trades: 0,
            wins: 0,
            losses: 0,
            win_pct: 0.0,
            profit_factor: 0.0,
            largest_win: 0.0,
            largest_loss: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            avg_trade: 0.0,
        };
    }

    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut wins: u32 = 0;
    let mut losses: u32 = 0;
    let mut largest_win = 0.0_f64;
    let mut largest_loss = 0.0_f64;
    let mut sum_wins = 0.0;
    let mut sum_losses = 0.0;

    for &pnl in pnls {
        if pnl > 0.0 {
            gross_profit += pnl;
            wins += 1;
            sum_wins += pnl;
            if pnl > largest_win {
                largest_win = pnl;
            }
        } else {
            gross_loss += pnl.abs();
            losses += 1;
            sum_losses += pnl;
            if pnl < largest_loss {
                largest_loss = pnl;
            }
        }
    }

    let total = pnls.len() as u32;
    let win_pct = if total > 0 {
        wins as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    TradeStatsResult {
        total_trades: total,
        wins,
        losses,
        win_pct,
        profit_factor,
        largest_win,
        largest_loss,
        avg_win: if wins > 0 { sum_wins / wins as f64 } else { 0.0 },
        avg_loss: if losses > 0 { sum_losses / losses as f64 } else { 0.0 },
        avg_trade: pnls.iter().sum::<f64>() / total as f64,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let returns = vec![0.10, -0.18182]; // 1.0 -> 1.1 -> 0.9
        let dd = compute_drawdown_full(&returns);
        assert!((dd.max_drawdown - 0.18182).abs() < 0.01);
    }

    #[test]
    fn profit_factor_calculation() {
        let pnls = vec![100.0, -50.0, 200.0, -30.0];
        let s = compute_stats(&[0.01; 4], &pnls, 0.0);
        assert!((s.profit_factor - 300.0 / 80.0).abs() < 0.01);
        assert_eq!(s.total_trades, 4);
        assert_eq!(s.win_rate, 0.5);
    }

    #[test]
    fn full_stats_empty() {
        let fs = compute_full_stats(&[], &[], &[], &[], 0, 0.0);
        assert_eq!(fs.total_return, 0.0);
        assert_eq!(fs.total_trades, 0);
    }

    #[test]
    fn full_stats_basic() {
        // 10 days of varying positive returns
        let daily = vec![0.01, 0.005, 0.02, -0.003, 0.015, 0.008, -0.002, 0.012, 0.007, 0.01];
        let mut capital = vec![100_000.0];
        for &r in &daily {
            capital.push(capital.last().unwrap() * (1.0 + r));
        }
        // Generate fake timestamps (2020-01-01 + daily)
        let base_ns: i64 = 1577836800_000_000_000; // 2020-01-01
        let ts: Vec<i64> = (0..capital.len())
            .map(|i| base_ns + i as i64 * 86_400_000_000_000)
            .collect();

        let fs = compute_full_stats(&capital, &ts, &[], &[], 0, 0.0);
        assert!(fs.total_return > 0.0);
        assert!(fs.volatility > 0.0);
        assert!(fs.daily.mean > 0.0);
    }

    #[test]
    fn full_stats_drawdown_avg() {
        // Up, crash, recover, crash again
        let returns = vec![0.10, -0.15, -0.05, 0.30, 0.05, -0.10, 0.20];
        let mut capital = vec![100_000.0];
        for &r in &returns {
            capital.push(capital.last().unwrap() * (1.0 + r));
        }
        let base_ns: i64 = 1577836800_000_000_000;
        let ts: Vec<i64> = (0..capital.len())
            .map(|i| base_ns + i as i64 * 86_400_000_000_000)
            .collect();

        let fs = compute_full_stats(&capital, &ts, &[], &[], 0, 0.0);
        assert!(fs.max_drawdown > 0.0);
        assert!(fs.avg_drawdown > 0.0);
        assert!(fs.avg_drawdown <= fs.max_drawdown);
    }

    #[test]
    fn full_stats_trade_pnls() {
        let capital = vec![100_000.0, 101_000.0, 102_000.0];
        let base_ns: i64 = 1577836800_000_000_000;
        let ts: Vec<i64> = (0..3).map(|i| base_ns + i * 86_400_000_000_000).collect();
        let pnls = vec![100.0, 200.0, -50.0];

        let fs = compute_full_stats(&capital, &ts, &pnls, &[], 0, 0.0);
        assert_eq!(fs.total_trades, 3);
        assert_eq!(fs.wins, 2);
        assert_eq!(fs.losses, 1);
        assert!((fs.profit_factor - 6.0).abs() < 0.01);
    }

    #[test]
    fn full_stats_turnover() {
        // 3 days, 2 stocks
        let weights = vec![
            0.5, 0.5, // day 0
            0.6, 0.4, // day 1: 0.1 change each
            0.6, 0.4, // day 2: no change
        ];
        let t = compute_turnover(&weights, 2);
        // day 1: sum(|0.1|+|0.1|)/2 = 0.1, day 2: 0 → avg = 0.05
        assert!((t - 0.05).abs() < 1e-10);
    }

    #[test]
    fn full_stats_herfindahl() {
        // 2 equal stocks → HHI = 0.5
        let weights = vec![0.5, 0.5, 0.5, 0.5];
        let h = compute_herfindahl(&weights, 2);
        assert!((h - 0.5).abs() < 1e-10);
    }

    #[test]
    fn percentile_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = percentile(&vals, 50.0);
        assert!((p50 - 5.5).abs() < 0.01);
        let p0 = percentile(&vals, 0.0);
        assert!((p0 - 1.0).abs() < 0.01);
        let p100 = percentile(&vals, 100.0);
        assert!((p100 - 10.0).abs() < 0.01);
    }

    #[test]
    fn days_to_ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        // 2020-01-01 = 18262 days since epoch
        let (y, m, d) = days_to_ymd(18262);
        assert_eq!((y, m, d), (2020, 1, 1));
    }

    #[test]
    fn ymd_roundtrip() {
        let ns = ymd_to_ns(2020, 6, 15);
        let days = (ns / 86_400_000_000_000) as i32;
        let (y, m, d) = days_to_ymd(days);
        assert_eq!((y, m, d), (2020, 6, 15));
    }

    #[test]
    fn lookback_basic() {
        // ~500 trading days from 2020-01-01
        let n = 500;
        let mut capital = vec![100_000.0];
        for i in 0..n {
            capital.push(capital[i] * 1.001); // small daily growth
        }
        let base_ns: i64 = ymd_to_ns(2020, 1, 1);
        let ts: Vec<i64> = (0..=n)
            .map(|i| base_ns + i as i64 * 86_400_000_000_000)
            .collect();

        let lb = compute_lookback(&capital, &ts);
        assert!(lb.mtd.is_some());
        assert!(lb.ytd.is_some());
        assert!(lb.one_year.is_some());
    }

    #[test]
    fn monthly_resample() {
        // Generate 90 days of data spanning ~3 months
        let n = 90;
        let mut capital = vec![100_000.0];
        for i in 0..n {
            capital.push(capital[i] * 1.001);
        }
        let base_ns: i64 = ymd_to_ns(2020, 1, 1);
        let ts: Vec<i64> = (0..=n)
            .map(|i| base_ns + i as i64 * 86_400_000_000_000)
            .collect();

        let monthly = resample_returns(&capital, &ts, ResampleFreq::Monthly);
        assert!(monthly.len() >= 2); // At least 2 monthly returns from 3 months
    }
}
