#!/usr/bin/env python3
"""Entry/exit timing analysis for tail-risk hedge trades.

Runs a configurable hedge strategy (SPY + puts with % of capital budget)
with monthly rebalancing so exit conditions are checked frequently.

Also analyzes market signals at each entry to see which conditions
predict profitable vs. unprofitable hedge trades — useful for building
a signal-based approach (only hedge when conditions warrant it).

Outputs:
  - Per-trade table with signal values at entry
  - Crash-period breakdown
  - Signal analysis (which signals correlated with better trades)
  - 6-panel chart saved to entry_exit_analysis.png
"""

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from backtester import Backtest, Stock, Type, Direction
from backtester.datahandler import HistoricalOptionsData, TiingoData
from backtester.strategy import Strategy, StrategyLeg
from backtester.enums import Order

# ---------------------------------------------------------------------------
# Strategy configuration — edit these to test different setups
# ---------------------------------------------------------------------------
CONFIG = {
    # Portfolio allocation
    'stock_alloc': 1.0,
    'options_alloc': 0.0,
    'cash_alloc': 0.0,
    'initial_capital': 1_000_000,

    # Options budget as % of total capital (evaluated each rebalance)
    'budget_pct': 0.1,  # 0.1% of capital per month

    # Rebalance frequency in months (1 = monthly, checks exits every month)
    'rebalance_months': 1,

    # Entry filters — max 4 months DTE
    'delta_min': -0.25,
    'delta_max': -0.10,
    'dte_min': 60,
    'dte_max': 120,

    # Exit filters — sell with ~1 month left
    'exit_dte': 30,

    # Profit/loss thresholds (math.inf = no cap)
    'profit_pct': math.inf,
    'loss_pct': math.inf,
}

CRASH_PERIODS = {
    '2008 GFC': (pd.Timestamp('2008-09-01'), pd.Timestamp('2008-11-30')),
    '2020 COVID': (pd.Timestamp('2020-02-01'), pd.Timestamp('2020-03-31')),
    '2022 Bear': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-10-31')),
}

# ---------------------------------------------------------------------------
# 1. Load data and compute signals
# ---------------------------------------------------------------------------
print("Loading data...")
options_data = HistoricalOptionsData("data/processed/options.csv")
stocks_data = TiingoData("data/processed/stocks.csv")
schema = options_data.schema
print(f"Date range: {stocks_data.start_date} to {stocks_data.end_date}")

# SPY price series for signal computation
spy_prices = stocks_data._data[stocks_data._data['symbol'] == 'SPY'].set_index('date')['adjClose'].sort_index()

# Compute market signals (all computable from our data, no external deps)
signals = pd.DataFrame(index=spy_prices.index)
signals['spy_price'] = spy_prices

# 1. Trailing 12-month return (high = expensive market, more need for hedge)
signals['ret_12m'] = spy_prices.pct_change(252) * 100

# 2. Trailing 6-month return
signals['ret_6m'] = spy_prices.pct_change(126) * 100

# 3. Distance from all-time high (0% = at ATH, -20% = bear territory)
signals['pct_from_ath'] = (spy_prices / spy_prices.cummax() - 1) * 100

# 4. Realized volatility (annualized, 30-day rolling)
signals['rvol_30d'] = spy_prices.pct_change().rolling(30).std() * np.sqrt(252) * 100

# 5. Average put implied vol from options data (monthly proxy for VIX)
opts = options_data._data
date_col = schema['date']
contract_col = schema['contract']
put_iv = opts[opts['type'] == 'put'].groupby('quotedate')['impliedvol'].mean()
signals['avg_put_iv'] = put_iv

signals = signals.ffill()

# ---------------------------------------------------------------------------
# 2. Run the backtest
# ---------------------------------------------------------------------------
cfg = CONFIG
print(f"\nStrategy config:")
print(f"  Budget: {cfg['budget_pct']}% of capital per rebalance")
print(f"  Delta: [{cfg['delta_min']}, {cfg['delta_max']}]")
print(f"  DTE: {cfg['dte_min']}-{cfg['dte_max']}, exit at DTE <= {cfg['exit_dte']}")
print(f"  Rebalance: every {cfg['rebalance_months']} month(s)")
profit_str = f"{cfg['profit_pct']}x" if cfg['profit_pct'] != math.inf else "none"
print(f"  Profit cap: {profit_str}")

leg = StrategyLeg('leg_1', schema, option_type=Type.PUT, direction=Direction.BUY)
leg.entry_filter = (
    (schema.underlying == 'SPY') &
    (schema.dte >= cfg['dte_min']) & (schema.dte <= cfg['dte_max']) &
    (schema.delta >= cfg['delta_min']) & (schema.delta <= cfg['delta_max'])
)
leg.entry_sort = ('delta', False)
leg.exit_filter = (schema.dte <= cfg['exit_dte'])

strategy = Strategy(schema)
strategy.add_leg(leg)
strategy.add_exit_thresholds(profit_pct=cfg['profit_pct'], loss_pct=cfg['loss_pct'])

bt = Backtest(
    {'stocks': cfg['stock_alloc'], 'options': cfg['options_alloc'], 'cash': cfg['cash_alloc']},
    initial_capital=cfg['initial_capital'],
)
budget_pct = cfg['budget_pct'] / 100.0
bt.options_budget = lambda date, total_capital: total_capital * budget_pct
bt.stocks = [Stock('SPY', 1.0)]
bt.stocks_data = stocks_data
bt.options_strategy = strategy
bt.options_data = options_data

print("\nRunning backtest...")
bt.run(rebalance_freq=cfg['rebalance_months'])
print(f"Done. {len(bt.trade_log)} trade log rows.\n")

# ---------------------------------------------------------------------------
# 3. Extract and match trades
# ---------------------------------------------------------------------------
tlog = bt.trade_log.copy()

entries = tlog[tlog[('leg_1', 'order')] == Order.BTO].copy()
exits = tlog[tlog[('leg_1', 'order')] == Order.STC].copy()


def flatten_cols(df):
    df = df.copy()
    df.columns = ['_'.join(str(c) for c in col).strip('_') for col in df.columns]
    return df


entries_flat = flatten_cols(entries)
exits_flat = flatten_cols(exits)

# Pair entries with exits by contract
trades = []
for _, entry in entries_flat.iterrows():
    contract = entry['leg_1_contract']
    entry_date = entry['totals_date']
    entry_cost_per = entry['leg_1_cost']  # ask * 100 (positive)
    entry_qty = entry['totals_qty']
    entry_strike = entry['leg_1_strike']

    premium_paid = entry_cost_per * entry_qty

    matching_exits = exits_flat[exits_flat['leg_1_contract'] == contract]
    if matching_exits.empty:
        trades.append({
            'contract': contract, 'strike': entry_strike,
            'entry_date': entry_date, 'exit_date': pd.NaT,
            'premium_paid': premium_paid, 'exit_value': 0.0,
            'qty': entry_qty, 'pnl': -premium_paid,
            'exit_reason': 'no_exit_found',
        })
        continue

    exit_row = matching_exits.iloc[0]
    exit_date = exit_row['totals_date']

    # Look up REAL bid from options data at exit date (backtester imputes wrong cost)
    exit_match = opts[
        (opts[contract_col] == contract) &
        (opts[date_col] == exit_date)
    ]
    if not exit_match.empty:
        real_bid = exit_match.iloc[0]['bid']
        exit_value = real_bid * 100 * entry_qty
        exit_reason = 'unknown'
    else:
        exit_value = 0.0
        exit_reason = 'expired_worthless'

    trades.append({
        'contract': contract, 'strike': entry_strike,
        'entry_date': entry_date, 'exit_date': exit_date,
        'premium_paid': premium_paid, 'exit_value': exit_value,
        'qty': entry_qty, 'pnl': exit_value - premium_paid,
        'exit_reason': exit_reason,
    })

trades_df = pd.DataFrame(trades)
trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
trades_df['holding_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
trades_df['pnl_pct'] = (trades_df['pnl'] / trades_df['premium_paid'] * 100).where(
    trades_df['premium_paid'] > 0
)

# ---------------------------------------------------------------------------
# 4. Look up greeks and classify exits
# ---------------------------------------------------------------------------
for idx, row in trades_df.iterrows():
    entry_match = opts[
        (opts[contract_col] == row['contract']) &
        (opts[date_col] == row['entry_date'])
    ]
    if not entry_match.empty:
        trades_df.at[idx, 'entry_dte'] = entry_match.iloc[0]['dte']
        trades_df.at[idx, 'entry_delta'] = entry_match.iloc[0]['delta']
        trades_df.at[idx, 'entry_iv'] = entry_match.iloc[0]['impliedvol']

    if pd.notna(row['exit_date']):
        exit_match = opts[
            (opts[contract_col] == row['contract']) &
            (opts[date_col] == row['exit_date'])
        ]
        if not exit_match.empty:
            trades_df.at[idx, 'exit_dte'] = exit_match.iloc[0]['dte']
            trades_df.at[idx, 'exit_delta'] = exit_match.iloc[0]['delta']

for idx, row in trades_df.iterrows():
    if row['exit_reason'] != 'unknown':
        continue
    if pd.notna(row.get('exit_dte')) and row['exit_dte'] <= cfg['exit_dte']:
        trades_df.at[idx, 'exit_reason'] = 'dte_exit'
    else:
        trades_df.at[idx, 'exit_reason'] = 'rebalance'

# ---------------------------------------------------------------------------
# 5. Attach signal values at entry date
# ---------------------------------------------------------------------------
for idx, row in trades_df.iterrows():
    entry_date = row['entry_date']
    if entry_date in signals.index:
        sig = signals.loc[entry_date]
    else:
        sig = signals.asof(entry_date)
    for col in ['ret_12m', 'ret_6m', 'pct_from_ath', 'rvol_30d', 'avg_put_iv']:
        trades_df.at[idx, col] = sig.get(col, np.nan)

# ---------------------------------------------------------------------------
# 6. Crash period analysis
# ---------------------------------------------------------------------------
def trades_active_during(df, start, end):
    mask = (df['entry_date'] <= end) & (
        df['exit_date'].isna() | (df['exit_date'] >= start)
    )
    return df[mask]

# ---------------------------------------------------------------------------
# 7. Console output
# ---------------------------------------------------------------------------
print("=" * 150)
print("PER-TRADE ANALYSIS")
print("=" * 150)
print(f"{'#':>3} {'Entry':>12} {'Exit':>12} {'Contract':<20} {'Strk':>6} {'DTE':>4} "
      f"{'Delta':>7} {'Hold':>5} {'Paid':>7} {'ExVal':>7} {'P&L':>9} {'P&L%':>7} "
      f"{'12mRet':>7} {'RVol':>5} {'PutIV':>6} {'Reason':<16}")
print("-" * 150)

for i, (_, row) in enumerate(trades_df.iterrows()):
    exit_str = row['exit_date'].strftime('%Y-%m-%d') if pd.notna(row['exit_date']) else 'N/A'
    entry_str = row['entry_date'].strftime('%Y-%m-%d')
    hold = f"{row['holding_days']:.0f}" if pd.notna(row['holding_days']) else 'N/A'
    dte = f"{row.get('entry_dte', 0):.0f}" if pd.notna(row.get('entry_dte')) else '?'
    delta = f"{row.get('entry_delta', 0):.2f}" if pd.notna(row.get('entry_delta')) else '?'
    pnl_pct = f"{row['pnl_pct']:.0f}%" if pd.notna(row['pnl_pct']) else 'N/A'
    ret12 = f"{row['ret_12m']:.0f}%" if pd.notna(row.get('ret_12m')) else '?'
    rvol = f"{row['rvol_30d']:.0f}" if pd.notna(row.get('rvol_30d')) else '?'
    piv = f"{row['avg_put_iv']:.2f}" if pd.notna(row.get('avg_put_iv')) else '?'
    print(f"{i+1:>3} {entry_str:>12} {exit_str:>12} {str(row['contract']):<20} "
          f"{row['strike']:>6.0f} {dte:>4} {delta:>7} {hold:>5} "
          f"${row['premium_paid']:>6,.0f} ${row['exit_value']:>6,.0f} "
          f"${row['pnl']:>8,.0f} {pnl_pct:>7} "
          f"{ret12:>7} {rvol:>5} {piv:>6} {row['exit_reason']:<16}")

# Crash breakdown
print("\n" + "=" * 80)
print("CRASH PERIOD BREAKDOWN")
print("=" * 80)
for name, (start, end) in CRASH_PERIODS.items():
    active = trades_active_during(trades_df, start, end)
    total_pnl = active['pnl'].sum()
    print(f"\n{name} ({start.strftime('%Y-%m')}-{end.strftime('%Y-%m')}):")
    if active.empty:
        print("  No active put positions during this period.")
    else:
        for _, row in active.iterrows():
            exit_str = row['exit_date'].strftime('%Y-%m-%d') if pd.notna(row['exit_date']) else 'N/A'
            print(f"  {row['contract']:<20} entry={row['entry_date'].strftime('%Y-%m-%d')} "
                  f"exit={exit_str} paid=${row['premium_paid']:,.0f} "
                  f"exit=${row['exit_value']:,.0f} P&L=${row['pnl']:,.0f}")
        print(f"  Total crash-period P&L: ${total_pnl:,.0f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
n_trades = len(trades_df)
winners = trades_df[trades_df['pnl'] > 0]
losers = trades_df[trades_df['pnl'] <= 0]
win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0
total_pnl = trades_df['pnl'].sum()
total_premium = trades_df['premium_paid'].sum()
avg_win = winners['pnl'].mean() if not winners.empty else 0
avg_loss = losers['pnl'].mean() if not losers.empty else 0
best_trade = trades_df.loc[trades_df['pnl'].idxmax()] if n_trades > 0 else None
worst_trade = trades_df.loc[trades_df['pnl'].idxmin()] if n_trades > 0 else None

years = (stocks_data.end_date - stocks_data.start_date).days / 365.25
cost_per_year = total_premium / years

print(f"Total trades:        {n_trades}")
print(f"Win rate:            {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
print(f"Total P&L from puts: ${total_pnl:,.0f}")
print(f"Total premium spent: ${total_premium:,.0f}")
print(f"Avg winning trade:   ${avg_win:,.0f}")
print(f"Avg losing trade:    ${avg_loss:,.0f}")
if best_trade is not None:
    print(f"Best trade:          ${best_trade['pnl']:,.0f} ({best_trade['contract']}, "
          f"{best_trade['entry_date'].strftime('%Y-%m-%d')})")
if worst_trade is not None:
    print(f"Worst trade:         ${worst_trade['pnl']:,.0f} ({worst_trade['contract']}, "
          f"{worst_trade['entry_date'].strftime('%Y-%m-%d')})")
print(f"Avg hedge cost/year: ${cost_per_year:,.0f}")

exit_reasons = trades_df['exit_reason'].value_counts()
print(f"\nExit reasons:")
for reason, count in exit_reasons.items():
    print(f"  {reason:<20} {count:>3} ({count/n_trades*100:.0f}%)")

# ---------------------------------------------------------------------------
# 8. Signal analysis — which conditions predicted profitable hedges?
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SIGNAL ANALYSIS")
print("=" * 80)

signal_cols = ['ret_12m', 'ret_6m', 'pct_from_ath', 'rvol_30d', 'avg_put_iv']
signal_names = {
    'ret_12m': '12-Month Return',
    'ret_6m': '6-Month Return',
    'pct_from_ath': '% From ATH',
    'rvol_30d': '30d Realized Vol',
    'avg_put_iv': 'Avg Put IV',
}

if not winners.empty and not losers.empty:
    print(f"\n{'Signal':<20} {'Avg (Winners)':>14} {'Avg (Losers)':>14} {'Direction':>12}")
    print("-" * 65)
    for col in signal_cols:
        w_avg = winners[col].mean()
        l_avg = losers[col].mean()
        direction = "higher" if w_avg > l_avg else "lower"
        print(f"{signal_names[col]:<20} {w_avg:>13.1f} {l_avg:>13.1f} {direction:>12}")
    print("\nInterpretation: signals where winners differ from losers suggest")
    print("conditions that favor hedging vs. staying fully in SPY.")
elif winners.empty:
    print("\nNo winning trades to compare signals against.")
    print("Signal values at entry for all trades:")
    print(f"\n{'Signal':<20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for col in signal_cols:
        vals = trades_df[col].dropna()
        if not vals.empty:
            print(f"{signal_names[col]:<20} {vals.mean():>10.1f} {vals.min():>10.1f} {vals.max():>10.1f}")

# Hypothetical: what if we only hedged when signal was above/below median?
print(f"\n--- Hypothetical signal filters ---")
for col in signal_cols:
    vals = trades_df[col].dropna()
    if vals.empty:
        continue
    median = vals.median()
    above = trades_df[trades_df[col] >= median]
    below = trades_df[trades_df[col] < median]
    pnl_above = above['pnl'].sum()
    pnl_below = below['pnl'].sum()
    premium_above = above['premium_paid'].sum()
    premium_below = below['premium_paid'].sum()
    print(f"\n{signal_names[col]} (median={median:.1f}):")
    print(f"  Above median: {len(above)} trades, P&L=${pnl_above:,.0f}, "
          f"premium=${premium_above:,.0f}, "
          f"eff={pnl_above/premium_above*100:.1f}%" if premium_above > 0 else "  Above median: 0 trades")
    print(f"  Below median: {len(below)} trades, P&L=${pnl_below:,.0f}, "
          f"premium=${premium_below:,.0f}, "
          f"eff={pnl_below/premium_below*100:.1f}%" if premium_below > 0 else "  Below median: 0 trades")

# ---------------------------------------------------------------------------
# 9. Charts
# ---------------------------------------------------------------------------
budget_label = f"{cfg['budget_pct']}% capital"

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle(
    f"Entry/Exit Analysis: SPY + Puts "
    f"(budget={budget_label}, delta=[{cfg['delta_min']},{cfg['delta_max']}], "
    f"DTE {cfg['dte_min']}-{cfg['dte_max']}, rebal {cfg['rebalance_months']}mo)",
    fontsize=12,
)

# -- Top-left: SPY price with entry/exit markers and crash shading --
ax = axes[0, 0]
ax.plot(spy_prices.index, spy_prices.values, color='black', linewidth=0.8, alpha=0.7, label='SPY')

crash_colors = {'2008 GFC': '#ff000030', '2020 COVID': '#ff660030', '2022 Bear': '#99000030'}
for name, (start, end) in CRASH_PERIODS.items():
    ax.axvspan(start, end, color=crash_colors[name], label=name)

for _, row in trades_df.dropna(subset=['entry_date']).iterrows():
    spy_at = spy_prices.asof(row['entry_date'])
    color = 'green' if row['pnl'] > 0 else 'red'
    ax.scatter(row['entry_date'], spy_at, marker='^', color=color, s=30, zorder=5, alpha=0.7)
for _, row in trades_df.dropna(subset=['exit_date']).iterrows():
    spy_at = spy_prices.asof(row['exit_date'])
    color = 'green' if row['pnl'] > 0 else 'red'
    ax.scatter(row['exit_date'], spy_at, marker='v', color=color, s=30, zorder=5, alpha=0.7)

ax.scatter([], [], marker='^', color='green', s=30, label='Entry (profit)')
ax.scatter([], [], marker='^', color='red', s=30, label='Entry (loss)')
ax.set_title('SPY Price with Put Entry/Exit Timing')
ax.set_ylabel('SPY Price ($)')
ax.legend(fontsize=7, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# -- Top-right: Per-trade P&L bar chart --
ax = axes[0, 1]
colors_bar = ['green' if p > 0 else 'red' for p in trades_df['pnl']]
ax.bar(range(len(trades_df)), trades_df['pnl'], color=colors_bar, alpha=0.7, width=1.0)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Per-Trade P&L (chronological)')
ax.set_xlabel('Trade #')
ax.set_ylabel('P&L ($)')

# -- Mid-left: Cumulative P&L from puts over time --
ax = axes[1, 0]
cum_pnl = trades_df.dropna(subset=['exit_date']).sort_values('exit_date')
if not cum_pnl.empty:
    cum_pnl_series = cum_pnl.set_index('exit_date')['pnl'].cumsum()
    ax.plot(cum_pnl_series.index, cum_pnl_series.values, color='steelblue', linewidth=1.5)
    ax.fill_between(cum_pnl_series.index, 0, cum_pnl_series.values,
                     where=cum_pnl_series.values >= 0, color='green', alpha=0.15)
    ax.fill_between(cum_pnl_series.index, 0, cum_pnl_series.values,
                     where=cum_pnl_series.values < 0, color='red', alpha=0.15)
    for name, (start, end) in CRASH_PERIODS.items():
        ax.axvspan(start, end, color=crash_colors[name], alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Cumulative P&L from Puts Over Time')
ax.set_ylabel('Cumulative P&L ($)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# -- Mid-right: P&L vs 12-month trailing return at entry --
ax = axes[1, 1]
valid = trades_df.dropna(subset=['ret_12m'])
if not valid.empty:
    colors_sc = ['green' if p > 0 else 'red' for p in valid['pnl']]
    ax.scatter(valid['ret_12m'], valid['pnl'], c=colors_sc, alpha=0.6, s=40)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax.set_title('Trade P&L vs 12-Month SPY Return at Entry')
ax.set_xlabel('Trailing 12-Month SPY Return (%)')
ax.set_ylabel('P&L ($)')

# -- Bottom-left: P&L vs realized vol at entry --
ax = axes[2, 0]
valid = trades_df.dropna(subset=['rvol_30d'])
if not valid.empty:
    colors_sc = ['green' if p > 0 else 'red' for p in valid['pnl']]
    ax.scatter(valid['rvol_30d'], valid['pnl'], c=colors_sc, alpha=0.6, s=40)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Trade P&L vs 30d Realized Vol at Entry')
ax.set_xlabel('Realized Volatility (%)')
ax.set_ylabel('P&L ($)')

# -- Bottom-right: Summary stats table --
ax = axes[2, 1]
ax.axis('off')
crash_pnl_rows = []
for name, (start, end) in CRASH_PERIODS.items():
    active = trades_active_during(trades_df, start, end)
    crash_pnl_rows.append([name, f"${active['pnl'].sum():,.0f}", str(len(active))])

table_data = [
    ['Total trades', str(n_trades), ''],
    ['Win rate', f'{win_rate:.1f}%', f'{len(winners)}W / {len(losers)}L'],
    ['Total put P&L', f'${total_pnl:,.0f}', ''],
    ['Total premium', f'${total_premium:,.0f}', ''],
    ['Avg win', f'${avg_win:,.0f}', ''],
    ['Avg loss', f'${avg_loss:,.0f}', ''],
    ['Hedge cost/year', f'${cost_per_year:,.0f}', ''],
    ['', '', ''],
    ['--- Crash P&L ---', 'P&L', 'Trades'],
] + crash_pnl_rows

table = ax.table(cellText=table_data,
                  colLabels=['Metric', 'Value', 'Detail'],
                  loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
for j in range(3):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Summary Statistics', pad=20)

plt.tight_layout()
plt.savefig('entry_exit_analysis.png', dpi=150)
print(f"\nSaved chart to entry_exit_analysis.png")
