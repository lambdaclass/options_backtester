/// Full backtest loop — mirrors BacktestEngine.run() for parity.
///
/// Pre-partitions all data by date at startup for O(1) lookups instead of
/// O(n) DataFrame scans on each access. Uses i64 nanosecond timestamps as
/// HashMap keys to avoid string conversion overhead entirely.
///
/// Key optimizations:
///   - filter_by_date()         → HashMap::get()          O(n) → O(1)
///   - get_contract_field_f64() → DayOptions::get_f64()   O(n) → O(1)
///   - get_contract_field_str() → DayOptions::get_str()   O(n) → O(1)
///   - get_symbol_price()       → DayStocks::get_price()  O(n) → O(1)
///   - Date keys are i64 (nanoseconds) — no string allocation or comparison.

use std::collections::HashMap;

use chrono::DateTime;
use polars::prelude::*;

use crate::cost_model::CostModel;
use crate::entries::compute_leg_entries;
use crate::fill_model::FillModel;
use crate::filter::CompiledFilter;
use crate::risk::{self, RiskConstraint};
use crate::signal_selector::SignalSelector;
use crate::stats;
use crate::types::{Direction, Greeks, LegConfig, Stats};

#[derive(Clone)]
pub struct BacktestConfig {
    pub allocation_stocks: f64,
    pub allocation_options: f64,
    pub allocation_cash: f64,
    pub initial_capital: f64,
    pub shares_per_contract: i64,
    pub legs: Vec<LegConfig>,
    pub profit_pct: Option<f64>,
    pub loss_pct: Option<f64>,
    pub stock_symbols: Vec<String>,
    pub stock_percentages: Vec<f64>,
    /// Pre-computed rebalance dates as nanoseconds since epoch.
    pub rebalance_dates: Vec<i64>,
    /// Transaction cost model.
    pub cost_model: CostModel,
    /// Fill model for execution pricing.
    pub fill_model: FillModel,
    /// Engine-level signal selector.
    pub signal_selector: SignalSelector,
    /// Risk constraints checked before entries.
    pub risk_constraints: Vec<RiskConstraint>,
    /// SMA days for stock gating (None = no SMA gate).
    pub sma_days: Option<usize>,
}

pub struct BacktestResult {
    pub balance: DataFrame,
    pub trade_log: DataFrame,
    pub final_cash: f64,
    pub stats: Stats,
}

struct Position {
    leg_contracts: Vec<String>,
    leg_types: Vec<String>,
    leg_directions: Vec<Direction>,
    quantity: f64,
    entry_cost: f64,
    greeks: Greeks,
}

struct StockHolding {
    symbol: String,
    qty: f64,
    price: f64,
}

/// Per-leg per-position entry in trade log (flat, converted to MultiIndex in Python).
struct TradeRow {
    date: i64,
    leg_data: Vec<LegTradeData>,
    total_cost: f64,
    qty: f64,
}

struct LegTradeData {
    contract: String,
    underlying: String,
    expiration: String,
    opt_type: String,
    strike: f64,
    cost: f64,
    order: String,
}

/// Balance row for a single date range day.
struct BalanceDay {
    date: i64,
    cash: f64,
    calls_capital: f64,
    puts_capital: f64,
    options_qty: f64,
    stocks_qty: f64,
    stock_values: Vec<(String, f64)>,
    stock_qtys: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// Date conversion helpers.
// ---------------------------------------------------------------------------

/// Convert nanoseconds since epoch to "YYYY-MM-DD HH:MM:SS" string.
fn ns_to_datestring(ns: i64) -> String {
    let secs = ns.div_euclid(1_000_000_000);
    let nsec = ns.rem_euclid(1_000_000_000) as u32;
    DateTime::from_timestamp(secs, nsec)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default()
}

/// Parse "YYYY-MM-DD HH:MM:SS" to nanoseconds since epoch.
fn parse_datestring_to_ns(s: &str) -> Option<i64> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .ok()
        .map(|dt| {
            let ts = dt.and_utc().timestamp();
            ts * 1_000_000_000
        })
}

/// Extract an i64 date key (nanoseconds) from a column value at index.
/// Handles Datetime (any time unit), Date, and String columns.
fn extract_date_ns(col: &Column, idx: usize) -> i64 {
    match col.dtype() {
        DataType::Datetime(tu, _) => {
            let val = col.datetime().unwrap().get(idx).unwrap_or(0);
            match tu {
                TimeUnit::Nanoseconds => val,
                TimeUnit::Microseconds => val * 1_000,
                TimeUnit::Milliseconds => val * 1_000_000,
            }
        }
        DataType::Date => {
            let days = col.date().unwrap().get(idx).unwrap_or(0);
            days as i64 * 86_400_000_000_000i64
        }
        _ => {
            col.str().ok()
                .and_then(|ca| ca.get(idx))
                .and_then(parse_datestring_to_ns)
                .unwrap_or(0)
        }
    }
}

/// Read a column value as a String, handling both String and Datetime columns.
fn column_value_to_string(col: &Column, idx: usize) -> String {
    if let Ok(ca) = col.str() {
        return ca.get(idx).unwrap_or("").to_string();
    }
    match col.dtype() {
        DataType::Datetime(tu, _) => {
            let val = col.datetime().unwrap().get(idx).unwrap_or(0);
            let ns = match tu {
                TimeUnit::Nanoseconds => val,
                TimeUnit::Microseconds => val * 1_000,
                TimeUnit::Milliseconds => val * 1_000_000,
            };
            ns_to_datestring(ns)
        }
        DataType::Date => {
            let days = col.date().unwrap().get(idx).unwrap_or(0);
            ns_to_datestring(days as i64 * 86_400_000_000_000i64)
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Pre-partitioned data structures — O(1) date and contract lookups.
// ---------------------------------------------------------------------------

/// Options data for a single date with O(1) contract lookups.
struct DayOptions {
    df: DataFrame,
    /// contract_string → row index within `df`.
    contract_idx: HashMap<String, usize>,
}

impl DayOptions {
    fn new(df: DataFrame, contract_col: &str) -> Self {
        let mut contract_idx = HashMap::new();
        if let Ok(col) = df.column(contract_col) {
            if let Ok(ca) = col.str() {
                for (i, val) in ca.into_iter().enumerate() {
                    if let Some(v) = val {
                        // Keep first occurrence (matches original filter + iloc[0]).
                        contract_idx.entry(v.to_string()).or_insert(i);
                    }
                }
            }
        }
        DayOptions { df, contract_idx }
    }

    /// Get a float64 field for a contract — O(1).
    fn get_f64(&self, contract: &str, field: &str) -> Option<f64> {
        let &row_idx = self.contract_idx.get(contract)?;
        let col = self.df.column(field).ok()?;
        // Fast path: column is already f64.
        if let Ok(ca) = col.f64() {
            return ca.get(row_idx);
        }
        // Slow path: cast to f64 (e.g. Int64 strike column).
        let casted = col.cast(&DataType::Float64).ok()?;
        casted.f64().ok()?.get(row_idx)
    }

    /// Get a string field for a contract — O(1).
    /// Handles both String and Datetime columns (for expiration).
    fn get_str(&self, contract: &str, field: &str) -> Option<String> {
        let &row_idx = self.contract_idx.get(contract)?;
        let col = self.df.column(field).ok()?;
        let s = column_value_to_string(col, row_idx);
        if s.is_empty() { None } else { Some(s) }
    }

    fn height(&self) -> usize {
        self.df.height()
    }
}

/// Stocks data for a single date — O(1) price lookups.
struct DayStocks {
    prices: HashMap<String, f64>,
}

impl DayStocks {
    fn get_price(&self, symbol: &str) -> Option<f64> {
        self.prices.get(symbol).copied()
    }
}

/// All data pre-partitioned by date.
struct PartitionedData {
    options: HashMap<i64, DayOptions>,
    stocks: HashMap<i64, DayStocks>,
    /// All option dates as nanoseconds, sorted ascending.
    all_dates_sorted: Vec<i64>,
}

/// Schema column name mappings passed from Python.
#[derive(Clone)]
pub struct SchemaMapping {
    pub contract: String,
    pub date: String,
    pub stocks_date: String,
    pub stocks_sym: String,
    pub stocks_price: String,
    pub underlying: String,
    pub expiration: String,
    pub option_type: String,
    pub strike: String,
}

// ---------------------------------------------------------------------------
// Main entry point.
// ---------------------------------------------------------------------------

pub fn run_backtest(
    config: &BacktestConfig,
    options_data: &DataFrame,
    stocks_data: &DataFrame,
    schema: &SchemaMapping,
) -> PolarsResult<BacktestResult> {
    let entry_filters: Vec<Option<CompiledFilter>> = config.legs.iter()
        .map(|leg| leg.entry_filter_query.as_ref().and_then(|q| CompiledFilter::new(q).ok()))
        .collect();
    let exit_filters: Vec<Option<CompiledFilter>> = config.legs.iter()
        .map(|leg| leg.exit_filter_query.as_ref().and_then(|q| CompiledFilter::new(q).ok()))
        .collect();

    let mut cash = config.initial_capital;
    let mut positions: Vec<Position> = Vec::new();
    let mut stock_holdings: Vec<StockHolding> = Vec::new();
    let mut peak_value: f64 = config.initial_capital;
    let mut portfolio_greeks = Greeks::default();

    let mut trade_rows: Vec<TradeRow> = Vec::new();
    let mut balance_days: Vec<BalanceDay> = Vec::new();

    // Pre-partition all data by date — single O(n) pass instead of
    // O(n) per-date filter calls throughout the backtest.
    let partitioned = prepartition_data(options_data, stocks_data, schema)?;

    // Pre-compute SMA per stock symbol if sma_days is set
    let sma_map_by_date = if let Some(sma_days) = config.sma_days {
        Some(compute_sma_map(&partitioned, &config.stock_symbols, sma_days))
    } else {
        None
    };

    let rb_dates = &config.rebalance_dates;
    if rb_dates.is_empty() {
        return build_result(&trade_rows, &balance_days, &config.legs, cash);
    }

    // Mirror Python: for each rebalance date, first compute balance since
    // previous rebalance date, then rebalance.
    for (rb_idx, &rb_date) in rb_dates.iter().enumerate() {
        let prev_rb_date = if rb_idx == 0 { rb_date } else { rb_dates[rb_idx - 1] };

        // _update_balance(prev_rb_date, rb_date)
        compute_balance_period(
            &positions, &stock_holdings,
            &partitioned,
            prev_rb_date, rb_date,
            config.shares_per_contract, cash,
            &config.legs,
            &mut balance_days,
        );

        // _rebalance_portfolio(rb_date, ...)
        let day_opts = match partitioned.options.get(&rb_date) {
            Some(d) => d,
            None => continue,
        };
        if day_opts.height() == 0 {
            continue;
        }
        let day_stocks = partitioned.stocks.get(&rb_date);

        // Execute exits (with cost model commission)
        execute_exits(
            &mut positions, &mut cash, day_opts,
            config.shares_per_contract,
            &config.legs, &exit_filters,
            config.profit_pct, config.loss_pct,
            schema, rb_date, &mut trade_rows,
            &config.cost_model,
        )?;

        // Compute total capital
        let stock_cap = compute_stock_capital(&stock_holdings, day_stocks);
        let options_cap = compute_options_capital(&positions, day_opts, config.shares_per_contract);
        let total_capital = cash + stock_cap + options_cap;

        // Track peak for drawdown risk checks
        peak_value = peak_value.max(total_capital);

        // Rebalance stocks
        let stocks_alloc = config.allocation_stocks * total_capital;
        stock_holdings.clear();
        cash = stocks_alloc + total_capital * config.allocation_cash;

        // Get SMA prices for this date if configured
        let sma_prices = sma_map_by_date.as_ref().and_then(|m| m.get(&rb_date));

        buy_stocks(
            &config.stock_symbols, &config.stock_percentages,
            day_stocks,
            stocks_alloc, &mut stock_holdings,
            &config.cost_model,
            &mut cash,
            sma_prices,
        );

        // Options rebalance
        let options_alloc = config.allocation_options * total_capital;
        if options_alloc >= options_cap {
            let budget = options_alloc - options_cap;
            cash += budget;

            let held: Vec<String> = positions.iter()
                .flat_map(|p| p.leg_contracts.clone())
                .collect();

            if let Some(pos) = execute_entries(
                &config.legs, &entry_filters, day_opts, &held,
                config.shares_per_contract, budget,
                schema, rb_date, &mut trade_rows,
                &config.fill_model, &config.signal_selector,
                &config.risk_constraints, &portfolio_greeks,
                total_capital, peak_value,
            )? {
                let cost = pos.entry_cost * pos.quantity;
                let commission = config.cost_model.option_cost(
                    cost.abs(), pos.quantity, config.shares_per_contract,
                );
                cash -= cost + commission;
                // Update portfolio greeks from entry
                portfolio_greeks += pos.greeks;
                positions.push(pos);
            }
        } else {
            // _sell_some_options
            let to_sell = options_cap - options_alloc;
            sell_some_options(
                &mut positions, &mut cash,
                day_opts, config.shares_per_contract,
                &config.legs, to_sell,
                schema, rb_date, &mut trade_rows,
            )?;
        }
    }

    // Final balance update: last rebalance date to end of data
    let last_rb = *rb_dates.last().unwrap();
    let last_date = partitioned.all_dates_sorted.last().copied().unwrap_or(0);

    if last_date > 0 {
        compute_balance_period(
            &positions, &stock_holdings,
            &partitioned,
            last_rb, last_date,
            config.shares_per_contract, cash,
            &config.legs,
            &mut balance_days,
        );
    }

    build_result(&trade_rows, &balance_days, &config.legs, cash)
}

// ---------------------------------------------------------------------------
// Data pre-partitioning — called once at startup.
// ---------------------------------------------------------------------------

fn prepartition_data(
    options_data: &DataFrame,
    stocks_data: &DataFrame,
    schema: &SchemaMapping,
) -> PolarsResult<PartitionedData> {
    let date_col = &schema.date;
    let contract_col = &schema.contract;

    // Sort options by date (skip if already sorted — common for CSV data).
    // slice() is zero-copy (shares underlying Arrow arrays via Arc).
    let date_series_raw = options_data.column(date_col)?;
    let n_check = options_data.height();
    let already_sorted = if n_check < 2 {
        true
    } else {
        let first = extract_date_ns(date_series_raw, 0);
        let last = extract_date_ns(date_series_raw, n_check - 1);
        if first > last {
            false
        } else {
            // Sample a few points to verify monotonicity cheaply.
            let step = (n_check / 8).max(1);
            let mut prev = first;
            let mut sorted = true;
            let mut i = step;
            while i < n_check {
                let val = extract_date_ns(date_series_raw, i);
                if val < prev {
                    sorted = false;
                    break;
                }
                prev = val;
                i += step;
            }
            sorted
        }
    };
    let sorted_opts;
    let date_series;
    if already_sorted {
        sorted_opts = options_data.clone();
        date_series = date_series_raw;
    } else {
        sorted_opts = options_data.sort([date_col.as_str()], SortMultipleOptions::default())?;
        date_series = sorted_opts.column(date_col)?;
    };
    let n_opts = sorted_opts.height();

    let mut options_map: HashMap<i64, DayOptions> = HashMap::new();
    let mut all_dates: Vec<i64> = Vec::new();

    if n_opts > 0 {
        let mut start = 0;
        let mut current = extract_date_ns(date_series, 0);

        for i in 1..n_opts {
            let d = extract_date_ns(date_series, i);
            if d != current {
                let part = sorted_opts.slice(start as i64, i - start);
                all_dates.push(current);
                options_map.insert(current, DayOptions::new(part, contract_col));
                current = d;
                start = i;
            }
        }
        // Last group
        let part = sorted_opts.slice(start as i64, n_opts - start);
        all_dates.push(current);
        options_map.insert(current, DayOptions::new(part, contract_col));
    }

    // Stocks: iterate once and build price HashMaps directly.
    // (Small data — typically 4500 rows, so no sort+slice needed.)
    let stocks_date_col = &schema.stocks_date;
    let sym_col_name = &schema.stocks_sym;
    let price_col_name = &schema.stocks_price;

    let stocks_date_series = stocks_data.column(stocks_date_col)?;
    let sym_ca = stocks_data.column(sym_col_name)?.str()?;
    let price_raw = stocks_data.column(price_col_name)?;
    let price_casted = price_raw.cast(&DataType::Float64)?;
    let price_ca = price_casted.f64()?;
    let n_stocks = stocks_data.height();

    let mut stocks_map: HashMap<i64, DayStocks> = HashMap::new();

    for i in 0..n_stocks {
        let date_ns = extract_date_ns(stocks_date_series, i);
        if let (Some(sym), Some(price)) = (sym_ca.get(i), price_ca.get(i)) {
            stocks_map.entry(date_ns)
                .or_insert_with(|| DayStocks { prices: HashMap::new() })
                .prices
                .insert(sym.to_string(), price);
        }
    }

    Ok(PartitionedData {
        options: options_map,
        stocks: stocks_map,
        all_dates_sorted: all_dates,
    })
}

// ---------------------------------------------------------------------------
// Execute exits.
// ---------------------------------------------------------------------------

fn execute_exits(
    positions: &mut Vec<Position>,
    cash: &mut f64,
    day_opts: &DayOptions,
    spc: i64,
    legs: &[LegConfig],
    exit_filters: &[Option<CompiledFilter>],
    profit_pct: Option<f64>,
    loss_pct: Option<f64>,
    schema: &SchemaMapping,
    date: i64,
    trade_rows: &mut Vec<TradeRow>,
    cost_model: &CostModel,
) -> PolarsResult<()> {
    let mut to_remove = Vec::new();

    for (i, pos) in positions.iter().enumerate() {
        let mut should_exit = false;

        // Check exit filters per leg
        for (j, _leg) in legs.iter().enumerate() {
            if let Some(ref flt) = exit_filters[j] {
                let contract = &pos.leg_contracts[j];
                if let Some(&row_idx) = day_opts.contract_idx.get(contract.as_str()) {
                    // Contract exists today — check exit filter on its row.
                    let one_row = day_opts.df.slice(row_idx as i64, 1);
                    if flt.apply(&one_row)?.height() > 0 {
                        should_exit = true;
                    }
                } else {
                    // Contract not in today's data → exit.
                    should_exit = true;
                }
            }
        }

        // Check threshold exits — mirrors Python's Strategy.filter_thresholds:
        //   excess_return = (current_cost / entry_cost + 1) * -sign(entry_cost)
        if !should_exit {
            let curr = compute_position_exit_cost(pos, day_opts, spc);
            let entry = pos.entry_cost;
            if entry != 0.0 {
                let excess_return = (curr / entry + 1.0) * -entry.signum();
                if profit_pct.map_or(false, |p| excess_return >= p)
                    || loss_pct.map_or(false, |l| excess_return <= -l)
                {
                    should_exit = true;
                }
            }
        }

        if should_exit {
            let exit_cost = compute_position_exit_cost(pos, day_opts, spc);
            *cash -= exit_cost * pos.quantity;

            // Apply exit commission
            let commission = cost_model.option_cost(
                exit_cost.abs(), pos.quantity.abs(), spc,
            );
            *cash -= commission;

            // Build trade row for exit
            let mut leg_data = Vec::new();
            for (j, leg) in legs.iter().enumerate() {
                let exit_price_col = leg.direction.invert().price_column();
                let price = day_opts.get_f64(&pos.leg_contracts[j], exit_price_col)
                    .unwrap_or(0.0);
                // Cash flow sign: BUY receives (-1), SELL pays (+1)
                let cash_sign = if leg.direction == Direction::Buy { -1.0 } else { 1.0 };
                let cost = cash_sign * price * spc as f64;
                let order = match leg.direction {
                    Direction::Buy => "STC",
                    Direction::Sell => "BTC",
                };
                leg_data.push(LegTradeData {
                    contract: pos.leg_contracts[j].clone(),
                    underlying: day_opts.get_str(&pos.leg_contracts[j], &schema.underlying)
                        .unwrap_or_default(),
                    expiration: day_opts.get_str(&pos.leg_contracts[j], &schema.expiration)
                        .unwrap_or_default(),
                    opt_type: pos.leg_types[j].clone(),
                    strike: day_opts.get_f64(&pos.leg_contracts[j], &schema.strike)
                        .unwrap_or(0.0),
                    cost,
                    order: order.to_string(),
                });
            }
            trade_rows.push(TradeRow {
                date,
                leg_data,
                total_cost: exit_cost,
                qty: pos.quantity,
            });
            to_remove.push(i);
        }
    }

    for &i in to_remove.iter().rev() {
        positions.remove(i);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sell some options (partial exit to reduce allocation).
// ---------------------------------------------------------------------------

fn sell_some_options(
    positions: &mut Vec<Position>,
    cash: &mut f64,
    day_opts: &DayOptions,
    spc: i64,
    legs: &[LegConfig],
    to_sell: f64,
    schema: &SchemaMapping,
    date: i64,
    trade_rows: &mut Vec<TradeRow>,
) -> PolarsResult<()> {
    let mut sold = 0.0;

    // Iterate positions (mirrors Python iterating inventory rows)
    let mut i = 0;
    while i < positions.len() {
        let pos = &positions[i];
        let exit_cost = compute_position_exit_cost(pos, day_opts, spc);

        if exit_cost == 0.0 {
            i += 1;
            continue;
        }

        if (to_sell - sold > -exit_cost) && (to_sell - sold > 0.0) {
            let mut qty_to_sell = ((to_sell - sold) / exit_cost).floor();
            if -qty_to_sell > pos.quantity {
                if qty_to_sell != 0.0 {
                    qty_to_sell = -pos.quantity;
                }
            }

            if qty_to_sell != 0.0 {
                // Build exit trade row
                let mut leg_data = Vec::new();
                for (j, leg) in legs.iter().enumerate() {
                    let exit_price_col = leg.direction.invert().price_column();
                    let price = day_opts.get_f64(&pos.leg_contracts[j], exit_price_col)
                        .unwrap_or(0.0);
                    let cash_sign = if leg.direction == Direction::Buy { -1.0 } else { 1.0 };
                    let cost = cash_sign * price * spc as f64;
                    let order = match leg.direction {
                        Direction::Buy => "STC",
                        Direction::Sell => "BTC",
                    };
                    leg_data.push(LegTradeData {
                        contract: pos.leg_contracts[j].clone(),
                        underlying: day_opts.get_str(&pos.leg_contracts[j], &schema.underlying)
                            .unwrap_or_default(),
                        expiration: day_opts.get_str(&pos.leg_contracts[j], &schema.expiration)
                            .unwrap_or_default(),
                        opt_type: pos.leg_types[j].clone(),
                        strike: day_opts.get_f64(&pos.leg_contracts[j], &schema.strike)
                            .unwrap_or(0.0),
                        cost,
                        order: order.to_string(),
                    });
                }
                trade_rows.push(TradeRow {
                    date,
                    leg_data,
                    total_cost: exit_cost,
                    qty: -qty_to_sell,
                });

                // Update position quantity
                let pos_mut = &mut positions[i];
                pos_mut.quantity += qty_to_sell;
                sold += qty_to_sell * exit_cost;

                // Remove position if fully sold
                if pos_mut.quantity <= 0.0 {
                    positions.remove(i);
                    continue;
                }
            }
        }
        i += 1;
    }

    *cash += sold - to_sell;
    Ok(())
}

// ---------------------------------------------------------------------------
// Execute entries.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn execute_entries(
    legs: &[LegConfig],
    entry_filters: &[Option<CompiledFilter>],
    day_opts: &DayOptions,
    held_contracts: &[String],
    spc: i64,
    budget: f64,
    schema: &SchemaMapping,
    date: i64,
    trade_rows: &mut Vec<TradeRow>,
    fill_model: &FillModel,
    signal_selector: &SignalSelector,
    risk_constraints: &[RiskConstraint],
    portfolio_greeks: &Greeks,
    total_capital: f64,
    peak_value: f64,
) -> PolarsResult<Option<Position>> {
    let contract_col = &schema.contract;
    if legs.is_empty() || budget <= 0.0 {
        return Ok(None);
    }

    // Determine extra columns needed by selectors
    let mut extra_cols: Vec<String> = Vec::new();
    for col_name in signal_selector.column_requirements() {
        extra_cols.push(col_name.to_string());
    }
    for leg in legs {
        if let Some(ref sel) = leg.signal_selector {
            for col_name in sel.column_requirements() {
                if !extra_cols.contains(&col_name.to_string()) {
                    extra_cols.push(col_name.to_string());
                }
            }
        }
    }

    let mut leg_results: Vec<DataFrame> = Vec::new();
    for (i, leg) in legs.iter().enumerate() {
        let filter = match &entry_filters[i] {
            Some(f) => f,
            None => return Ok(None),
        };
        let entries = compute_leg_entries(
            &day_opts.df, held_contracts, filter, contract_col,
            leg.direction.price_column(),
            leg.entry_sort_col.as_deref(), leg.entry_sort_asc,
            spc, leg.direction == Direction::Sell,
        )?;
        if entries.height() == 0 {
            return Ok(None);
        }
        leg_results.push(entries);
    }

    // Apply signal selector per leg to pick the best row
    let mut leg_contracts = Vec::new();
    let mut leg_types = Vec::new();
    let mut leg_directions = Vec::new();
    let mut total_cost = 0.0;
    let mut leg_data = Vec::new();
    let mut entry_greeks = Greeks::default();

    for (i, leg_df) in leg_results.iter().enumerate() {
        // Per-leg selector override, or engine-level selector
        let sel = legs[i].signal_selector.as_ref().unwrap_or(signal_selector);
        let row_idx = sel.select_index(leg_df);

        let contract = leg_df.column("contract")?.str()?.get(row_idx).unwrap_or("").to_string();
        let opt_type = leg_df.column("type")?.str()?.get(row_idx).unwrap_or("").to_string();
        let mut cost = leg_df.column("cost")?.f64()?.get(row_idx).unwrap_or(0.0);
        let underlying = leg_df.column("underlying")?.str()?.get(row_idx).unwrap_or("").to_string();
        // Handle expiration as either String or Datetime
        let expiration = column_value_to_string(leg_df.column("expiration")?, row_idx);
        let strike = leg_df.column("strike")?.f64()?.get(row_idx).unwrap_or(0.0);

        // Apply fill model to re-price if not MarketAtBidAsk
        let leg_fill = legs[i].fill_model.as_ref().unwrap_or(fill_model);
        if !matches!(leg_fill, FillModel::MarketAtBidAsk) {
            // Look up bid/ask/volume from day data for this contract
            if let (Some(bid), Some(ask)) = (
                day_opts.get_f64(&contract, "bid"),
                day_opts.get_f64(&contract, "ask"),
            ) {
                let volume = day_opts.get_f64(&contract, "volume");
                let is_buy = legs[i].direction == Direction::Buy;
                let fill_price = leg_fill.fill_price(bid, ask, volume, is_buy);
                let sign = if is_buy { 1.0 } else { -1.0 };
                cost = sign * fill_price * spc as f64;
            }
        }

        // Collect Greeks from the entry row (for risk checking)
        let dir_sign = if legs[i].direction == Direction::Buy { 1.0 } else { -1.0 };
        let delta = day_opts.get_f64(&contract, "delta").unwrap_or(0.0);
        let gamma = day_opts.get_f64(&contract, "gamma").unwrap_or(0.0);
        let theta = day_opts.get_f64(&contract, "theta").unwrap_or(0.0);
        let vega = day_opts.get_f64(&contract, "vega").unwrap_or(0.0);

        let order = match legs[i].direction {
            Direction::Buy => "BTO",
            Direction::Sell => "STO",
        };

        leg_data.push(LegTradeData {
            contract: contract.clone(),
            underlying,
            expiration,
            opt_type: opt_type.clone(),
            strike,
            cost,
            order: order.to_string(),
        });

        leg_contracts.push(contract);
        leg_types.push(opt_type);
        leg_directions.push(legs[i].direction);
        total_cost += cost;

        // Accumulate greeks (will be scaled by qty later)
        entry_greeks.delta += delta * dir_sign;
        entry_greeks.gamma += gamma * dir_sign;
        entry_greeks.theta += theta * dir_sign;
        entry_greeks.vega += vega * dir_sign;
    }

    if total_cost.abs() == 0.0 {
        return Ok(None);
    }

    let qty = (budget / total_cost.abs()).floor();
    if qty <= 0.0 {
        return Ok(None);
    }

    // Scale greeks by quantity
    let scaled_greeks = entry_greeks.scale(qty);

    // Risk check: reject entry if any constraint fails
    if !risk_constraints.is_empty() {
        if !risk::check_all(risk_constraints, portfolio_greeks, &scaled_greeks, total_capital, peak_value) {
            return Ok(None);
        }
    }

    trade_rows.push(TradeRow {
        date,
        leg_data,
        total_cost,
        qty,
    });

    Ok(Some(Position {
        leg_contracts,
        leg_types,
        leg_directions,
        quantity: qty,
        entry_cost: total_cost,
        greeks: scaled_greeks,
    }))
}

// ---------------------------------------------------------------------------
// Compute balance for a date range — uses pre-partitioned data.
// ---------------------------------------------------------------------------

fn compute_balance_period(
    positions: &[Position],
    stock_holdings: &[StockHolding],
    partitioned: &PartitionedData,
    start_date: i64,
    end_date: i64,
    spc: i64,
    cash: f64,
    legs: &[LegConfig],
    balance_days: &mut Vec<BalanceDay>,
) {
    // Binary search for dates in [start_date, end_date).
    let dates = &partitioned.all_dates_sorted;
    let start_idx = dates.partition_point(|&d| d < start_date);
    let end_idx = dates.partition_point(|&d| d < end_date);

    for &d in &dates[start_idx..end_idx] {
        let day_opts = partitioned.options.get(&d);
        let day_stocks = partitioned.stocks.get(&d);

        // Compute calls/puts capital for each position
        let mut calls_cap = 0.0;
        let mut puts_cap = 0.0;
        let mut options_qty = 0.0;

        if let Some(opts) = day_opts {
            for pos in positions {
                options_qty += pos.quantity;
                for (j, leg) in legs.iter().enumerate() {
                    if j >= pos.leg_contracts.len() { continue; }
                    let exit_price_col = leg.direction.invert().price_column();
                    let price = opts.get_f64(&pos.leg_contracts[j], exit_price_col)
                        .unwrap_or(0.0);
                    let sign = leg.direction.invert().sign();
                    let value = sign * price * pos.quantity * spc as f64;

                    if pos.leg_types[j] == "call" {
                        calls_cap += value;
                    } else {
                        puts_cap += value;
                    }
                }
            }
        }

        // Compute stock values
        let mut stock_values = Vec::new();
        let mut stock_qtys = Vec::new();
        let mut stocks_qty = 0.0;
        for holding in stock_holdings {
            let price = day_stocks
                .and_then(|ds| ds.get_price(&holding.symbol))
                .unwrap_or(holding.price);
            stock_values.push((holding.symbol.clone(), holding.qty * price));
            stock_qtys.push((holding.symbol.clone(), holding.qty));
            stocks_qty += holding.qty;
        }

        balance_days.push(BalanceDay {
            date: d,
            cash,
            calls_capital: calls_cap,
            puts_capital: puts_cap,
            options_qty,
            stocks_qty,
            stock_values,
            stock_qtys,
        });
    }
}

// ---------------------------------------------------------------------------
// SMA computation — uses pre-partitioned stocks data.
// ---------------------------------------------------------------------------

fn compute_sma_map(
    partitioned: &PartitionedData,
    symbols: &[String],
    sma_days: usize,
) -> HashMap<i64, HashMap<String, f64>> {
    let mut result: HashMap<i64, HashMap<String, f64>> = HashMap::new();

    for symbol in symbols {
        // Collect (date_ns, price) pairs for this symbol from pre-partitioned data.
        let mut date_prices: Vec<(i64, f64)> = Vec::new();
        for &date_ns in &partitioned.all_dates_sorted {
            if let Some(ds) = partitioned.stocks.get(&date_ns) {
                if let Some(price) = ds.get_price(symbol) {
                    date_prices.push((date_ns, price));
                }
            }
        }

        // Compute rolling SMA
        for (i, &(date_ns, _)) in date_prices.iter().enumerate() {
            if i + 1 < sma_days {
                continue; // Not enough data yet
            }
            let start = i + 1 - sma_days;
            let sum: f64 = date_prices[start..=i].iter().map(|&(_, p)| p).sum();
            let sma = sum / sma_days as f64;

            result.entry(date_ns)
                .or_default()
                .insert(symbol.clone(), sma);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Build result DataFrames.
// ---------------------------------------------------------------------------

fn build_result(
    trade_rows: &[TradeRow],
    balance_days: &[BalanceDay],
    legs: &[LegConfig],
    final_cash: f64,
) -> PolarsResult<BacktestResult> {
    // Build trade log as flat DataFrame (Python converts to MultiIndex)
    let n_trades = trade_rows.len();
    let mut trade_dates: Vec<String> = Vec::with_capacity(n_trades);
    let mut trade_total_costs: Vec<f64> = Vec::with_capacity(n_trades);
    let mut trade_qtys: Vec<f64> = Vec::with_capacity(n_trades);

    // Per-leg columns
    let mut leg_columns: Vec<Vec<(String, String, String, String, f64, f64, String)>> =
        legs.iter().map(|_| Vec::with_capacity(n_trades)).collect();

    for tr in trade_rows {
        trade_dates.push(ns_to_datestring(tr.date));
        trade_total_costs.push(tr.total_cost);
        trade_qtys.push(tr.qty);
        for (j, ld) in tr.leg_data.iter().enumerate() {
            if j < leg_columns.len() {
                leg_columns[j].push((
                    ld.contract.clone(), ld.underlying.clone(), ld.expiration.clone(),
                    ld.opt_type.clone(), ld.strike, ld.cost, ld.order.clone(),
                ));
            }
        }
    }

    let mut trade_cols: Vec<Column> = vec![
        Column::new("totals__date".into(), &trade_dates),
        Column::new("totals__cost".into(), &trade_total_costs),
        Column::new("totals__qty".into(), &trade_qtys),
    ];
    for (j, leg) in legs.iter().enumerate() {
        if j < leg_columns.len() {
            let data = &leg_columns[j];
            let prefix = &leg.name;
            trade_cols.push(Column::new(format!("{prefix}__contract").into(),
                data.iter().map(|d| d.0.as_str()).collect::<Vec<_>>()));
            trade_cols.push(Column::new(format!("{prefix}__underlying").into(),
                data.iter().map(|d| d.1.as_str()).collect::<Vec<_>>()));
            trade_cols.push(Column::new(format!("{prefix}__expiration").into(),
                data.iter().map(|d| d.2.as_str()).collect::<Vec<_>>()));
            trade_cols.push(Column::new(format!("{prefix}__type").into(),
                data.iter().map(|d| d.3.as_str()).collect::<Vec<_>>()));
            trade_cols.push(Column::new(format!("{prefix}__strike").into(),
                data.iter().map(|d| d.4).collect::<Vec<_>>()));
            trade_cols.push(Column::new(format!("{prefix}__cost").into(),
                data.iter().map(|d| d.5).collect::<Vec<_>>()));
            trade_cols.push(Column::new(format!("{prefix}__order").into(),
                data.iter().map(|d| d.6.as_str()).collect::<Vec<_>>()));
        }
    }
    let trade_log = DataFrame::new(trade_cols)?;

    // Build balance DataFrame
    let n_days = balance_days.len();
    let mut bal_dates: Vec<String> = Vec::with_capacity(n_days);
    let mut bal_cash: Vec<f64> = Vec::with_capacity(n_days);
    let mut bal_calls: Vec<f64> = Vec::with_capacity(n_days);
    let mut bal_puts: Vec<f64> = Vec::with_capacity(n_days);
    let mut bal_opts_qty: Vec<f64> = Vec::with_capacity(n_days);
    let mut bal_stocks_qty: Vec<f64> = Vec::with_capacity(n_days);

    // Collect all stock symbols
    let mut stock_symbols: Vec<String> = Vec::new();
    if let Some(first) = balance_days.first() {
        stock_symbols = first.stock_values.iter().map(|(s, _)| s.clone()).collect();
    }
    let mut stock_val_cols: Vec<Vec<f64>> = stock_symbols.iter().map(|_| Vec::with_capacity(n_days)).collect();
    let mut stock_qty_cols: Vec<Vec<f64>> = stock_symbols.iter().map(|_| Vec::with_capacity(n_days)).collect();

    for day in balance_days {
        bal_dates.push(ns_to_datestring(day.date));
        bal_cash.push(day.cash);
        bal_calls.push(day.calls_capital);
        bal_puts.push(day.puts_capital);
        bal_opts_qty.push(day.options_qty);
        bal_stocks_qty.push(day.stocks_qty);
        for (k, sym) in stock_symbols.iter().enumerate() {
            let val = day.stock_values.iter().find(|(s, _)| s == sym).map(|(_, v)| *v).unwrap_or(0.0);
            let qty = day.stock_qtys.iter().find(|(s, _)| s == sym).map(|(_, q)| *q).unwrap_or(0.0);
            stock_val_cols[k].push(val);
            stock_qty_cols[k].push(qty);
        }
    }

    let mut bal_cols: Vec<Column> = vec![
        Column::new("date".into(), &bal_dates),
        Column::new("cash".into(), &bal_cash),
        Column::new("calls capital".into(), &bal_calls),
        Column::new("puts capital".into(), &bal_puts),
        Column::new("options qty".into(), &bal_opts_qty),
        Column::new("stocks qty".into(), &bal_stocks_qty),
    ];
    for (k, sym) in stock_symbols.iter().enumerate() {
        bal_cols.push(Column::new(sym.as_str().into(), &stock_val_cols[k]));
        bal_cols.push(Column::new(format!("{sym} qty").into(), &stock_qty_cols[k]));
    }
    let balance = DataFrame::new(bal_cols)?;

    // Stats from balance
    let totals: Vec<f64> = balance_days.iter().map(|d| {
        let stock_val: f64 = d.stock_values.iter().map(|(_, v)| *v).sum();
        d.cash + d.calls_capital + d.puts_capital + stock_val
    }).collect();
    let daily_returns = compute_daily_returns(&totals);
    let result_stats = stats::compute_stats(&daily_returns, &[], 0.0);

    Ok(BacktestResult { balance, trade_log, final_cash, stats: result_stats })
}

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

/// Compute the total exit cost for a position — O(1) per leg via DayOptions.
fn compute_position_exit_cost(pos: &Position, day_opts: &DayOptions, spc: i64) -> f64 {
    let mut total = 0.0;
    for (i, contract) in pos.leg_contracts.iter().enumerate() {
        let dir = pos.leg_directions[i];
        let price = day_opts.get_f64(contract, dir.invert().price_column())
            .unwrap_or(0.0);
        let cash_sign = if dir == Direction::Buy { -1.0 } else { 1.0 };
        total += cash_sign * price * spc as f64;
    }
    total
}

fn compute_stock_capital(holdings: &[StockHolding], day_stocks: Option<&DayStocks>) -> f64 {
    let ds = match day_stocks {
        Some(ds) => ds,
        None => return 0.0,
    };
    holdings.iter().map(|h| {
        ds.get_price(&h.symbol).unwrap_or(0.0) * h.qty
    }).sum()
}

fn compute_options_capital(
    positions: &[Position], day_opts: &DayOptions, spc: i64,
) -> f64 {
    positions.iter().map(|pos| {
        -compute_position_exit_cost(pos, day_opts, spc) * pos.quantity
    }).sum()
}

fn buy_stocks(
    symbols: &[String], percentages: &[f64],
    day_stocks: Option<&DayStocks>,
    allocation: f64, holdings: &mut Vec<StockHolding>,
    cost_model: &CostModel,
    cash: &mut f64,
    sma_prices: Option<&HashMap<String, f64>>,
) {
    let ds = match day_stocks {
        Some(ds) => ds,
        None => return,
    };
    let mut stock_cost_total = 0.0;
    let mut commission_total = 0.0;
    for (symbol, pct) in symbols.iter().zip(percentages) {
        if let Some(price) = ds.get_price(symbol) {
            if price > 0.0 {
                // SMA gating: only buy if sma < price
                if let Some(sma_map) = sma_prices {
                    if let Some(&sma_val) = sma_map.get(symbol) {
                        if sma_val >= price {
                            holdings.push(StockHolding { symbol: symbol.clone(), qty: 0.0, price });
                            continue;
                        }
                    }
                }
                let qty = (allocation * pct / price).floor();
                commission_total += cost_model.stock_cost(price, qty);
                stock_cost_total += qty * price;
                holdings.push(StockHolding { symbol: symbol.clone(), qty, price });
            }
        }
    }
    *cash -= stock_cost_total + commission_total;
}

fn compute_daily_returns(totals: &[f64]) -> Vec<f64> {
    if totals.len() < 2 { return Vec::new(); }
    totals.windows(2).map(|w| if w[0] != 0.0 { (w[1] - w[0]) / w[0] } else { 0.0 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_returns_basic() {
        let totals = vec![100.0, 110.0, 105.0];
        let returns = compute_daily_returns(&totals);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10);
        assert!((returns[1] - (-5.0 / 110.0)).abs() < 1e-10);
    }

    #[test]
    fn daily_returns_empty() {
        assert!(compute_daily_returns(&[]).is_empty());
        assert!(compute_daily_returns(&[100.0]).is_empty());
    }

    #[test]
    fn ns_to_datestring_epoch() {
        assert_eq!(ns_to_datestring(0), "1970-01-01 00:00:00");
    }

    #[test]
    fn ns_roundtrip() {
        let s = "2024-06-15 00:00:00";
        let ns = parse_datestring_to_ns(s).unwrap();
        assert_eq!(ns_to_datestring(ns), s);
    }
}
