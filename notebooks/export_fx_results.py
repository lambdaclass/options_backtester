"""Export FX carry backtest capital series to CSV.

The Databento CME data is proprietary, so the results notebook loads
pre-computed capital series instead of raw data.

Usage:  python notebooks/export_fx_results.py
"""
import os, sys, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'databento')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exported')
os.makedirs(OUT_DIR, exist_ok=True)

MONTH_CODES = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,
               'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}

# ── Inline all functions from fx_carry_real.ipynb ──

def load_front_month(filename):
    fut = pd.read_parquet(os.path.join(DATA_DIR, filename))
    outrights = fut[~fut['symbol'].str.contains('-', na=False)].copy().sort_index()
    contract_prices = {}
    for _, row in outrights.iterrows():
        sym, date = row['symbol'], row.name.normalize().tz_localize(None)
        contract_prices.setdefault(sym, {})[date] = row['close']
    daily_front = {}
    for date, group in outrights.groupby(outrights.index.date):
        best = group.sort_values('volume', ascending=False).iloc[0]
        daily_front[pd.Timestamp(date)] = {'symbol': best['symbol'], 'close': best['close'], 'volume': best['volume']}
    dates = sorted(daily_front.keys())
    records, prev_date, prev_symbol = [], None, None
    for date in dates:
        info = daily_front[date]
        cur_symbol = info['symbol']
        if prev_date is None:
            records.append({'date': date, 'close': info['close'], 'return': 0.0})
        elif cur_symbol == prev_symbol:
            pp = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            cp = contract_prices.get(cur_symbol, {}).get(date, 0)
            records.append({'date': date, 'close': info['close'], 'return': cp/pp - 1 if pp > 0 else 0.0})
        else:
            op = contract_prices.get(prev_symbol, {}).get(prev_date, 0)
            oc = contract_prices.get(prev_symbol, {}).get(date, 0)
            records.append({'date': date, 'close': info['close'], 'return': oc/op - 1 if op > 0 and oc > 0 else 0.0})
        prev_date, prev_symbol = date, cur_symbol
    return pd.DataFrame(records).set_index('date')

def parse_option(sym, date_year, product='AUD'):
    parts = sym.split()
    if len(parts) != 2: return None
    contract, opt = parts
    opt_type = opt[0]
    if opt_type not in ('C', 'P'): return None
    try: strike_raw = int(opt[1:])
    except ValueError: return None
    if product == 'AUD':
        strike = strike_raw / 1000.0
        if contract.startswith('ADU'): mc, yd = contract[3], int(contract[4])
        elif contract.startswith('6A'): mc, yd = contract[2], int(contract[3])
        else: return None
    elif product == 'JPY':
        strike = strike_raw / 100000.0
        if contract.startswith('JPU'): mc, yd = contract[3], int(contract[4])
        elif contract.startswith('6J'): mc, yd = contract[2], int(contract[3])
        else: return None
    else: return None
    month = MONTH_CODES.get(mc, 0)
    if month == 0: return None
    year = (date_year // 10) * 10 + yd
    if year < date_year - 2: year += 10
    return month, year, opt_type, strike

def load_fx_options(old_file, new_file, product, cutoff='2016-08-23'):
    old = pd.read_parquet(os.path.join(DATA_DIR, old_file))
    new = pd.read_parquet(os.path.join(DATA_DIR, new_file))
    old = old[~old['symbol'].str.contains('UD:', na=False)].copy()
    new = new[~new['symbol'].str.contains('UD:', na=False)].copy()
    old = old[old.index < pd.Timestamp(cutoff, tz='UTC')]
    combined = pd.concat([old, new]).sort_index()
    records = []
    for idx, row in combined.iterrows():
        parsed = parse_option(row['symbol'], idx.year, product=product)
        if parsed is None: continue
        month, year, opt_type, strike = parsed
        fom = pd.Timestamp(year=year, month=month, day=1)
        third_wed = fom + pd.offsets.WeekOfMonth(week=2, weekday=2)
        expiry = (third_wed - pd.offsets.BDay(2)).tz_localize('UTC')
        records.append({'date': idx, 'symbol': row['symbol'], 'opt_type': opt_type,
                        'strike': strike, 'expiry': expiry, 'close': row['close'], 'volume': row['volume']})
    return pd.DataFrame(records)

def select_monthly_options(opts, front_prices, opt_type='P', otm_target=0.92):
    filtered = opts[opts['opt_type'] == opt_type].copy()
    prices = front_prices[['close']].rename(columns={'close': 'fut_close'})
    prices.index = prices.index.tz_localize('UTC')
    filtered['date_norm'] = filtered['date'].dt.normalize()
    filtered = filtered.merge(prices, left_on='date_norm', right_index=True, how='left').dropna(subset=['fut_close'])
    filtered['moneyness'] = filtered['strike'] / filtered['fut_close']
    filtered['year_month'] = filtered['date'].dt.to_period('M')
    selections = []
    for ym, group in filtered.groupby('year_month'):
        first_day = group['date'].min()
        day_opts = group[(group['date'] == first_day) & (group['expiry'] > first_day + pd.Timedelta(days=14))]
        if len(day_opts) == 0: continue
        day_opts = day_opts[day_opts['expiry'] == day_opts['expiry'].min()]
        day_opts = day_opts[day_opts['moneyness'] < 1.0] if opt_type == 'P' else day_opts[day_opts['moneyness'] > 1.0]
        if len(day_opts) == 0: continue
        day_opts = day_opts.copy()
        day_opts['dist'] = (day_opts['moneyness'] - otm_target).abs()
        best = day_opts.nsmallest(5, 'dist').sort_values('volume', ascending=False).iloc[0]
        if best['close'] <= 0: continue
        selections.append({'entry_date': first_day, 'symbol': best['symbol'], 'strike': best['strike'],
                           'entry_price': best['close'], 'expiry': best['expiry'],
                           'underlying': best['fut_close'], 'moneyness': best['moneyness'], 'volume': best['volume']})
    return pd.DataFrame(selections)

def build_settlement_lookup(opts):
    lookup = {}
    for _, row in opts.iterrows():
        lookup.setdefault(row['symbol'], []).append((row['date'], row['close']))
    for sym in lookup: lookup[sym].sort(key=lambda x: x[0])
    return lookup

def get_settlement(symbol, strike, expiry, opt_type, lookup, front_prices):
    ws, we = expiry - pd.Timedelta(days=5), expiry + pd.Timedelta(days=2)
    if symbol in lookup:
        near = [(d, p) for d, p in lookup[symbol] if ws <= d <= we]
        if near: return near[-1][1]
    nd = front_prices[(front_prices.index >= (expiry - pd.Timedelta(days=3)).tz_localize(None)) &
                       (front_prices.index <= (expiry + pd.Timedelta(days=3)).tz_localize(None))]
    if len(nd) > 0:
        u = nd.iloc[-1]['close']
        return max(0, strike - u) if opt_type == 'P' else max(0, u - strike)
    return 0.0

def precompute_settlements(selections, opt_type, lookup, front_prices):
    m = {}
    for _, row in selections.iterrows():
        settle = get_settlement(row['symbol'], row['strike'], row['expiry'], opt_type, lookup, front_prices)
        ep = row['entry_price']
        m[row['entry_date']] = {'pnl_ratio': (settle - ep) / ep if ep > 0 else 0}
    return m

def run_backtest(cross, aud_front, jpy_front, aud_put_sels, jpy_call_sels,
                 aud_opts, jpy_opts, leverage=1, aud_budget=0.005, jpy_budget=0.0):
    aud_lookup = build_settlement_lookup(aud_opts) if aud_budget > 0 else {}
    jpy_lookup = build_settlement_lookup(jpy_opts) if jpy_budget > 0 else {}
    aud_map = precompute_settlements(aud_put_sels, 'P', aud_lookup, aud_front) if aud_budget > 0 else {}
    jpy_map = precompute_settlements(jpy_call_sels, 'C', jpy_lookup, jpy_front) if jpy_budget > 0 else {}
    capital, records, current_month = 100.0, [], None
    for date in cross.index:
        if capital <= 0:
            records.append({'date': date, 'capital': 0}); continue
        notional = capital * leverage
        carry = notional * cross.loc[date, 'daily_carry']
        spot = notional * cross.loc[date, 'cross_ret']
        aud_pnl = jpy_pnl = 0
        ym = pd.Timestamp(date).to_period('M')
        if ym != current_month:
            current_month = ym
            dtz = pd.Timestamp(date, tz='UTC')
            if aud_budget > 0 and dtz in aud_map:
                aud_pnl = aud_budget * notional * aud_map[dtz]['pnl_ratio']
            if jpy_budget > 0 and dtz in jpy_map:
                jpy_pnl = jpy_budget * notional * jpy_map[dtz]['pnl_ratio']
        capital += carry + spot + aud_pnl + jpy_pnl
        records.append({'date': date, 'capital': capital})
    return pd.DataFrame(records).set_index('date')

# ── Run ──
print('Loading futures...')
aud = load_front_month('6A_FUT_ohlcv1d.parquet')
jpy = load_front_month('6J_FUT_ohlcv1d.parquet')

common = aud.index.intersection(jpy.index)
rba = {2010:4.25,2011:4.50,2012:3.50,2013:2.75,2014:2.50,2015:2.00,
       2016:1.75,2017:1.50,2018:1.50,2019:1.00,2020:0.25,2021:0.10,
       2022:1.85,2023:4.10,2024:4.35,2025:4.35}
boj = {y:0.0 for y in range(2010,2026)}; boj[2024]=0.25; boj[2025]=0.50

cross = pd.DataFrame({
    'audjpy': aud.loc[common,'close'] / jpy.loc[common,'close'],
    'cross_ret': aud.loc[common,'return'] - jpy.loc[common,'return'],
})
cross['daily_carry'] = (cross.index.year.map(lambda y: rba.get(y,0)) -
                         cross.index.year.map(lambda y: boj.get(y,0))) / 36500

print('Loading options...')
aud_opts = load_fx_options('6A_OPT_ohlcv1d.parquet', 'ADU_OPT_ohlcv1d.parquet', 'AUD')
jpy_opts = load_fx_options('6J_OPT_ohlcv1d.parquet', 'JPU_OPT_ohlcv1d.parquet', 'JPY', cutoff='2016-08-16')

aud_front = pd.DataFrame({'close': aud['close']})
jpy_front = pd.DataFrame({'close': jpy['close']})
aud_puts_8 = select_monthly_options(aud_opts, aud_front, opt_type='P', otm_target=0.92)
jpy_calls_8 = select_monthly_options(jpy_opts, jpy_front, opt_type='C', otm_target=1.08)
empty = pd.DataFrame(columns=['entry_date','symbol','strike','entry_price','expiry','underlying','moneyness','volume'])

configs = {
    '1x_unhedged': (1, empty, empty, 0, 0),
    '1x_aud_puts': (1, aud_puts_8, empty, 0.005, 0),
    '1x_jpy_calls': (1, empty, jpy_calls_8, 0, 0.005),
    '1x_dual_hedge': (1, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
    '3x_unhedged': (3, empty, empty, 0, 0),
    '3x_aud_puts': (3, aud_puts_8, empty, 0.005, 0),
    '3x_jpy_calls': (3, empty, jpy_calls_8, 0, 0.005),
    '3x_dual_hedge': (3, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
    '5x_unhedged': (5, empty, empty, 0, 0),
    '5x_aud_puts': (5, aud_puts_8, empty, 0.005, 0),
    '5x_jpy_calls': (5, empty, jpy_calls_8, 0, 0.005),
    '5x_dual_hedge': (5, aud_puts_8, jpy_calls_8, 0.0025, 0.0025),
}

capital_df = pd.DataFrame()
for label, (lev, a_s, j_s, ab, jb) in configs.items():
    print(f'  {label}...')
    res = run_backtest(cross, aud_front, jpy_front, a_s, j_s, aud_opts, jpy_opts, leverage=lev, aud_budget=ab, jpy_budget=jb)
    capital_df[label] = res['capital']
capital_df['audjpy'] = cross['audjpy']

out = os.path.join(OUT_DIR, 'fx_carry_capital.csv')
capital_df.to_csv(out)
print(f'\nSaved {out} ({len(capital_df)} rows, {len(capital_df.columns)} cols)')
