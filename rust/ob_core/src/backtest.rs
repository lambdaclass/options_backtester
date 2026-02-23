/// Full backtest loop — mirrors BacktestEngine.run() for parity.
///
/// Accepts pre-computed rebalance dates from Python (avoids reimplementing
/// pandas BMS logic). Iterates all dates, computes balance for date ranges
/// between rebalances, executes exits/entries/partial sells identically.

use polars::prelude::*;

use crate::entries::compute_leg_entries;
use crate::filter::CompiledFilter;
use crate::stats;
use crate::types::{Direction, LegConfig, Stats};

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
    /// Pre-computed rebalance date strings (from Python's BMS grouper).
    pub rebalance_dates: Vec<String>,
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
    leg_costs: Vec<f64>,
    quantity: f64,
    entry_cost: f64,
}

struct StockHolding {
    symbol: String,
    qty: f64,
    price: f64,
}

/// Per-leg per-position entry in trade log (flat, converted to MultiIndex in Python).
struct TradeRow {
    date: String,
    // Per-leg data: leg_name -> (contract, underlying, expiration, type, strike, cost, order)
    leg_data: Vec<LegTradeData>,
    total_cost: f64,
    qty: f64,
}

struct LegTradeData {
    leg_name: String,
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
    date: String,
    cash: f64,
    calls_capital: f64,
    puts_capital: f64,
    options_qty: f64,
    stocks_qty: f64,
    stock_values: Vec<(String, f64)>,
    stock_qtys: Vec<(String, f64)>,
}

pub fn run_backtest(
    config: &BacktestConfig,
    options_data: &DataFrame,
    stocks_data: &DataFrame,
    contract_col: &str,
    date_col: &str,
    stocks_date_col: &str,
    stocks_sym_col: &str,
    stocks_price_col: &str,
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

    let mut trade_rows: Vec<TradeRow> = Vec::new();
    let mut balance_days: Vec<BalanceDay> = Vec::new();

    let rb_dates = &config.rebalance_dates;
    if rb_dates.is_empty() {
        return build_result(&trade_rows, &balance_days, &config.legs, cash);
    }

    // Mirror Python: for each rebalance date, first compute balance since
    // previous rebalance date, then rebalance.
    for (rb_idx, rb_date) in rb_dates.iter().enumerate() {
        let prev_rb_date = if rb_idx == 0 { rb_date } else { &rb_dates[rb_idx - 1] };

        // _update_balance(prev_rb_date, rb_date)
        compute_balance_period(
            &positions, &stock_holdings,
            options_data, stocks_data,
            contract_col, date_col, stocks_date_col, stocks_sym_col, stocks_price_col,
            prev_rb_date, rb_date,
            config.shares_per_contract, cash,
            &config.legs,
            &mut balance_days,
        )?;

        // _rebalance_portfolio(rb_date, ...)
        let day_opts = filter_by_date(options_data, date_col, rb_date)?;
        let day_stocks = filter_by_date(stocks_data, stocks_date_col, rb_date)?;
        if day_opts.height() == 0 {
            continue;
        }

        // Execute exits
        execute_exits(
            &mut positions, &mut cash, &day_opts,
            contract_col, config.shares_per_contract,
            &config.legs, &exit_filters,
            config.profit_pct, config.loss_pct,
            schema, rb_date, &mut trade_rows,
        )?;

        // Compute total capital
        let stock_cap = compute_stock_capital(&stock_holdings, &day_stocks, stocks_sym_col, stocks_price_col);
        let options_cap = compute_options_capital(&positions, &day_opts, contract_col, config.shares_per_contract);
        let total_capital = cash + stock_cap + options_cap;

        // Rebalance stocks
        let stocks_alloc = config.allocation_stocks * total_capital;
        stock_holdings.clear();
        cash = stocks_alloc + total_capital * config.allocation_cash;
        buy_stocks(
            &config.stock_symbols, &config.stock_percentages,
            &day_stocks, stocks_sym_col, stocks_price_col,
            stocks_alloc, &mut stock_holdings,
        );
        let stock_cost: f64 = stock_holdings.iter().map(|h| h.qty * h.price).sum();
        cash -= stock_cost;

        // Options rebalance
        let options_alloc = config.allocation_options * total_capital;
        if options_alloc >= options_cap {
            let budget = options_alloc - options_cap;
            cash += budget;

            let held: Vec<String> = positions.iter()
                .flat_map(|p| p.leg_contracts.clone())
                .collect();

            if let Some(pos) = execute_entries(
                &config.legs, &entry_filters, &day_opts, &held,
                contract_col, config.shares_per_contract, budget,
                schema, rb_date, &mut trade_rows,
            )? {
                let cost = pos.entry_cost * pos.quantity;
                cash -= cost;
                positions.push(pos);
            }
        } else {
            // _sell_some_options
            let to_sell = options_cap - options_alloc;
            sell_some_options(
                &mut positions, &mut cash,
                &day_opts, contract_col, config.shares_per_contract,
                &config.legs, to_sell,
                schema, rb_date, &mut trade_rows,
            )?;
        }
    }

    // Final balance update: last rebalance date to end of data
    let last_rb = rb_dates.last().unwrap();
    let all_dates_col = options_data.column(date_col)?.unique()?.sort(Default::default())?;
    let all_dates = all_dates_col.as_materialized_series();
    let last_date = series_to_strings(all_dates)?.last().cloned().unwrap_or_default();

    if !last_date.is_empty() {
        compute_balance_period(
            &positions, &stock_holdings,
            options_data, stocks_data,
            contract_col, date_col, stocks_date_col, stocks_sym_col, stocks_price_col,
            last_rb, &last_date,
            config.shares_per_contract, cash,
            &config.legs,
            &mut balance_days,
        )?;
    }

    build_result(&trade_rows, &balance_days, &config.legs, cash)
}

/// Schema column name mappings passed from Python.
pub struct SchemaMapping {
    pub underlying: String,
    pub expiration: String,
    pub option_type: String,
    pub strike: String,
}

fn execute_exits(
    positions: &mut Vec<Position>,
    cash: &mut f64,
    day_opts: &DataFrame,
    contract_col: &str,
    spc: i64,
    legs: &[LegConfig],
    exit_filters: &[Option<CompiledFilter>],
    profit_pct: Option<f64>,
    loss_pct: Option<f64>,
    schema: &SchemaMapping,
    date: &str,
    trade_rows: &mut Vec<TradeRow>,
) -> PolarsResult<()> {
    let mut to_remove = Vec::new();

    for (i, pos) in positions.iter().enumerate() {
        let mut should_exit = false;

        // Check exit filters per leg
        for (j, _leg) in legs.iter().enumerate() {
            if let Some(ref flt) = exit_filters[j] {
                let contract = &pos.leg_contracts[j];
                let matched = day_opts.clone().lazy()
                    .filter(col(contract_col).eq(lit(contract.as_str())))
                    .collect()?;
                if matched.height() == 0 {
                    should_exit = true;
                } else if flt.apply(&matched)?.height() > 0 {
                    should_exit = true;
                }
            }
        }

        // Check threshold exits
        if !should_exit {
            let curr = compute_position_exit_cost(pos, day_opts, contract_col, spc)?;
            let entry = pos.entry_cost;
            if entry != 0.0 {
                let pnl_pct = (curr - entry) / entry.abs();
                if profit_pct.map_or(false, |p| pnl_pct >= p)
                    || loss_pct.map_or(false, |l| pnl_pct <= -l)
                {
                    should_exit = true;
                }
            }
        }

        if should_exit {
            let exit_cost = compute_position_exit_cost(pos, day_opts, contract_col, spc).unwrap_or(0.0);
            *cash -= exit_cost * pos.quantity;

            // Build trade row for exit
            let mut leg_data = Vec::new();
            for (j, leg) in legs.iter().enumerate() {
                let exit_price_col = leg.direction.invert().price_column();
                let price = get_contract_field_f64(
                    day_opts, contract_col, &pos.leg_contracts[j], exit_price_col,
                ).unwrap_or(0.0);
                let sign = leg.direction.invert().sign();
                let cost = sign * price * spc as f64;
                let order = match leg.direction {
                    Direction::Buy => "STC",
                    Direction::Sell => "BTC",
                };
                leg_data.push(LegTradeData {
                    leg_name: leg.name.clone(),
                    contract: pos.leg_contracts[j].clone(),
                    underlying: get_contract_field_str(day_opts, contract_col, &pos.leg_contracts[j], &schema.underlying)
                        .unwrap_or_default(),
                    expiration: get_contract_field_str(day_opts, contract_col, &pos.leg_contracts[j], &schema.expiration)
                        .unwrap_or_default(),
                    opt_type: pos.leg_types[j].clone(),
                    strike: get_contract_field_f64(day_opts, contract_col, &pos.leg_contracts[j], &schema.strike)
                        .unwrap_or(0.0),
                    cost,
                    order: order.to_string(),
                });
            }
            trade_rows.push(TradeRow {
                date: date.to_string(),
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

fn sell_some_options(
    positions: &mut Vec<Position>,
    cash: &mut f64,
    day_opts: &DataFrame,
    contract_col: &str,
    spc: i64,
    legs: &[LegConfig],
    to_sell: f64,
    schema: &SchemaMapping,
    date: &str,
    trade_rows: &mut Vec<TradeRow>,
) -> PolarsResult<()> {
    let mut sold = 0.0;

    // Iterate positions (mirrors Python iterating inventory rows)
    let mut i = 0;
    while i < positions.len() {
        let pos = &positions[i];
        let exit_cost = compute_position_exit_cost(pos, day_opts, contract_col, spc)?;

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
                    let price = get_contract_field_f64(
                        day_opts, contract_col, &pos.leg_contracts[j], exit_price_col,
                    ).unwrap_or(0.0);
                    let sign = leg.direction.invert().sign();
                    let cost = sign * price * spc as f64;
                    let order = match leg.direction {
                        Direction::Buy => "STC",
                        Direction::Sell => "BTC",
                    };
                    leg_data.push(LegTradeData {
                        leg_name: leg.name.clone(),
                        contract: pos.leg_contracts[j].clone(),
                        underlying: get_contract_field_str(day_opts, contract_col, &pos.leg_contracts[j], &schema.underlying)
                            .unwrap_or_default(),
                        expiration: get_contract_field_str(day_opts, contract_col, &pos.leg_contracts[j], &schema.expiration)
                            .unwrap_or_default(),
                        opt_type: pos.leg_types[j].clone(),
                        strike: get_contract_field_f64(day_opts, contract_col, &pos.leg_contracts[j], &schema.strike)
                            .unwrap_or(0.0),
                        cost,
                        order: order.to_string(),
                    });
                }
                trade_rows.push(TradeRow {
                    date: date.to_string(),
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

fn execute_entries(
    legs: &[LegConfig],
    entry_filters: &[Option<CompiledFilter>],
    day_opts: &DataFrame,
    held_contracts: &[String],
    contract_col: &str,
    spc: i64,
    budget: f64,
    schema: &SchemaMapping,
    date: &str,
    trade_rows: &mut Vec<TradeRow>,
) -> PolarsResult<Option<Position>> {
    if legs.is_empty() || budget <= 0.0 {
        return Ok(None);
    }

    let mut leg_results: Vec<DataFrame> = Vec::new();
    for (i, leg) in legs.iter().enumerate() {
        let filter = match &entry_filters[i] {
            Some(f) => f,
            None => return Ok(None),
        };
        let entries = compute_leg_entries(
            day_opts, held_contracts, filter, contract_col,
            leg.direction.price_column(),
            leg.entry_sort_col.as_deref(), leg.entry_sort_asc,
            spc, leg.direction == Direction::Sell,
        )?;
        if entries.height() == 0 {
            return Ok(None);
        }
        leg_results.push(entries);
    }

    // Take first row from each leg (FirstMatch selector)
    let mut leg_contracts = Vec::new();
    let mut leg_types = Vec::new();
    let mut leg_directions = Vec::new();
    let mut leg_costs = Vec::new();
    let mut total_cost = 0.0;
    let mut leg_data = Vec::new();

    for (i, leg_df) in leg_results.iter().enumerate() {
        let contract = leg_df.column("contract")?.str()?.get(0).unwrap_or("").to_string();
        let opt_type = leg_df.column("type")?.str()?.get(0).unwrap_or("").to_string();
        let cost = leg_df.column("cost")?.f64()?.get(0).unwrap_or(0.0);
        let underlying = leg_df.column("underlying")?.str()?.get(0).unwrap_or("").to_string();
        let expiration = leg_df.column("expiration")?.str()?.get(0).unwrap_or("").to_string();
        let strike = leg_df.column("strike")?.f64()?.get(0).unwrap_or(0.0);

        let order = match legs[i].direction {
            Direction::Buy => "BTO",
            Direction::Sell => "STO",
        };

        leg_data.push(LegTradeData {
            leg_name: legs[i].name.clone(),
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
        leg_costs.push(cost);
        total_cost += cost;
    }

    if total_cost.abs() == 0.0 {
        return Ok(None);
    }

    let qty = (budget / total_cost.abs()).floor();
    if qty <= 0.0 {
        return Ok(None);
    }

    trade_rows.push(TradeRow {
        date: date.to_string(),
        leg_data,
        total_cost,
        qty,
    });

    Ok(Some(Position {
        leg_contracts,
        leg_types,
        leg_directions,
        leg_costs,
        quantity: qty,
        entry_cost: total_cost,
    }))
}

/// Compute balance for all dates in [start_date, end_date) — mirrors _update_balance.
fn compute_balance_period(
    positions: &[Position],
    stock_holdings: &[StockHolding],
    options_data: &DataFrame,
    stocks_data: &DataFrame,
    contract_col: &str,
    date_col: &str,
    stocks_date_col: &str,
    stocks_sym_col: &str,
    stocks_price_col: &str,
    start_date: &str,
    end_date: &str,
    spc: i64,
    cash: f64,
    legs: &[LegConfig],
    balance_days: &mut Vec<BalanceDay>,
) -> PolarsResult<()> {
    // Get all dates in [start_date, end_date)
    let period_opts = options_data.clone().lazy()
        .filter(col(date_col).gt_eq(lit(start_date)).and(col(date_col).lt(lit(end_date))))
        .collect()?;
    let period_stocks = stocks_data.clone().lazy()
        .filter(col(stocks_date_col).gt_eq(lit(start_date)).and(col(stocks_date_col).lt(lit(end_date))))
        .collect()?;

    let unique_dates = period_opts.column(date_col)?.unique()?.sort(Default::default())?;
    let dates = column_to_strings(&unique_dates)?;

    for d in &dates {
        let day_opts = filter_by_date(&period_opts, date_col, d)?;
        let day_stocks = filter_by_date(&period_stocks, stocks_date_col, d)?;

        // Compute calls/puts capital for each position
        let mut calls_cap = 0.0;
        let mut puts_cap = 0.0;
        let mut options_qty = 0.0;

        for pos in positions {
            options_qty += pos.quantity;
            for (j, leg) in legs.iter().enumerate() {
                if j >= pos.leg_contracts.len() { continue; }
                let exit_price_col = leg.direction.invert().price_column();
                let price = get_contract_field_f64(
                    &day_opts, contract_col, &pos.leg_contracts[j], exit_price_col,
                ).unwrap_or(0.0);
                let sign = leg.direction.invert().sign();
                let value = sign * price * pos.quantity * spc as f64;

                if pos.leg_types[j] == "call" {
                    calls_cap += value;
                } else {
                    puts_cap += value;
                }
            }
        }

        // Compute stock values
        let mut stock_values = Vec::new();
        let mut stock_qtys = Vec::new();
        let mut stocks_qty = 0.0;
        for holding in stock_holdings {
            let price = get_symbol_price(&day_stocks, stocks_sym_col, stocks_price_col, &holding.symbol)
                .unwrap_or(holding.price);
            stock_values.push((holding.symbol.clone(), holding.qty * price));
            stock_qtys.push((holding.symbol.clone(), holding.qty));
            stocks_qty += holding.qty;
        }

        balance_days.push(BalanceDay {
            date: d.clone(),
            cash,
            calls_capital: calls_cap,
            puts_capital: puts_cap,
            options_qty,
            stocks_qty,
            stock_values,
            stock_qtys,
        });
    }
    Ok(())
}

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
        trade_dates.push(tr.date.clone());
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
        bal_dates.push(day.date.clone());
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

// ---------- helpers ----------

fn filter_by_date(df: &DataFrame, date_col: &str, date: &str) -> PolarsResult<DataFrame> {
    df.clone().lazy().filter(col(date_col).eq(lit(date))).collect()
}

fn series_to_strings(s: &Series) -> PolarsResult<Vec<String>> {
    let ca = s.str()?;
    Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
}

fn column_to_strings(c: &Column) -> PolarsResult<Vec<String>> {
    series_to_strings(c.as_materialized_series())
}

fn get_contract_field_f64(
    df: &DataFrame, contract_col: &str, contract: &str, field: &str,
) -> Option<f64> {
    let matched = df.clone().lazy()
        .filter(col(contract_col).eq(lit(contract)))
        .select([col(field)])
        .collect().ok()?;
    if matched.height() == 0 { return None; }
    matched.column(field).ok()?.f64().ok()?.get(0)
}

fn get_contract_field_str(
    df: &DataFrame, contract_col: &str, contract: &str, field: &str,
) -> Option<String> {
    let matched = df.clone().lazy()
        .filter(col(contract_col).eq(lit(contract)))
        .select([col(field)])
        .collect().ok()?;
    if matched.height() == 0 { return None; }
    matched.column(field).ok()?.str().ok()?.get(0).map(|s| s.to_string())
}

fn get_symbol_price(
    df: &DataFrame, sym_col: &str, price_col: &str, symbol: &str,
) -> Option<f64> {
    let matched = df.clone().lazy()
        .filter(col(sym_col).eq(lit(symbol)))
        .select([col(price_col)])
        .collect().ok()?;
    if matched.height() == 0 { return None; }
    matched.column(price_col).ok()?.f64().ok()?.get(0)
}

fn compute_position_exit_cost(
    pos: &Position, day_opts: &DataFrame, contract_col: &str, spc: i64,
) -> PolarsResult<f64> {
    let mut total = 0.0;
    for (i, contract) in pos.leg_contracts.iter().enumerate() {
        let dir = pos.leg_directions[i];
        let price = get_contract_field_f64(day_opts, contract_col, contract, dir.invert().price_column())
            .unwrap_or(0.0);
        total += dir.invert().sign() * price * spc as f64;
    }
    Ok(total)
}

fn compute_stock_capital(
    holdings: &[StockHolding], day_stocks: &DataFrame, sym_col: &str, price_col: &str,
) -> f64 {
    holdings.iter().map(|h| {
        get_symbol_price(day_stocks, sym_col, price_col, &h.symbol).unwrap_or(0.0) * h.qty
    }).sum()
}

fn compute_options_capital(
    positions: &[Position], day_opts: &DataFrame, contract_col: &str, spc: i64,
) -> f64 {
    positions.iter().map(|pos| {
        -compute_position_exit_cost(pos, day_opts, contract_col, spc).unwrap_or(0.0) * pos.quantity
    }).sum()
}

fn buy_stocks(
    symbols: &[String], percentages: &[f64],
    day_stocks: &DataFrame, sym_col: &str, price_col: &str,
    allocation: f64, holdings: &mut Vec<StockHolding>,
) {
    for (symbol, pct) in symbols.iter().zip(percentages) {
        if let Some(price) = get_symbol_price(day_stocks, sym_col, price_col, symbol) {
            if price > 0.0 {
                let qty = (allocation * pct / price).floor();
                holdings.push(StockHolding { symbol: symbol.clone(), qty, price });
            }
        }
    }
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
}
