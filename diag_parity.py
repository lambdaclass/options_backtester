import os, sys
sys.path.insert(0, '.')
from tests.bench._parity_helpers import *
import options_portfolio_backtester.engine._dispatch as _d

print("RUST_AVAILABLE:", _d.RUST_AVAILABLE)
print("use_rust():", _d.use_rust())
print()

# ===================================================
# 1. MidPrice fill model
# ===================================================
from options_portfolio_backtester.execution.fill_model import MidPrice
alloc = DEFAULT_ALLOC
cap = DEFAULT_CAPITAL

_d.RUST_AVAILABLE = False
py = run_python(alloc, cap, buy_put_strategy, fill_model=MidPrice())
_d.RUST_AVAILABLE = True

rs = run_rust(alloc, cap, buy_put_strategy, fill_model=MidPrice())

print("=== MIDPRICE ===")
print("Python trade_log shape:", py.trade_log.shape)
print("Rust trade_log shape:", rs.trade_log.shape)
print("\nPython trade_log:")
print(py.trade_log.to_string())
print("\nRust trade_log:")
print(rs.trade_log.to_string())
print("\nPython balance (first 5):")
print(py.balance.head().to_string())
print("\nRust balance (first 5):")
print(rs.balance.head().to_string())
print("\nPython final capital:", py.balance["total capital"].iloc[-1])
print("Rust final capital:", rs.balance["total capital"].iloc[-1])

# ===================================================
# 2. NearestDelta
# ===================================================
from options_portfolio_backtester.execution.signal_selector import NearestDelta
ss = NearestDelta(target_delta=-0.30)

_d.RUST_AVAILABLE = False
py = run_python(alloc, cap, buy_put_strategy, signal_selector=ss)
_d.RUST_AVAILABLE = True
rs = run_rust(alloc, cap, buy_put_strategy, signal_selector=ss)

print("\n=== NEAREST DELTA ===")
print("Python trade_log:")
print(py.trade_log.to_string())
print("\nRust trade_log:")
print(rs.trade_log.to_string())

# ===================================================
# 3. Multi-leg (strangle)
# ===================================================
def strat(schema):
    return two_leg_strategy(schema, "buy", "call", "buy", "put")

_d.RUST_AVAILABLE = False
py = run_python(alloc, cap, strat)
_d.RUST_AVAILABLE = True
rs = run_rust(alloc, cap, strat)

print("\n=== BUY STRANGLE ===")
print("Python trade_log shape:", py.trade_log.shape)
print("Rust trade_log shape:", rs.trade_log.shape)
print("\nPython trade_log:")
print(py.trade_log.to_string())
print("\nRust trade_log:")
print(rs.trade_log.to_string())

# ===================================================
# 4. SELL PUT and BUY CALL
# ===================================================
_d.RUST_AVAILABLE = False
py_sell = run_python(alloc, cap, sell_put_strategy)
_d.RUST_AVAILABLE = True
rs_sell = run_rust(alloc, cap, sell_put_strategy)

print("\n=== SELL PUT ===")
print("Python trade_log:")
print(py_sell.trade_log.to_string())
print("\nRust trade_log:")
print(rs_sell.trade_log.to_string())
print("Python final:", py_sell.balance["total capital"].iloc[-1])
print("Rust final:", rs_sell.balance["total capital"].iloc[-1])

_d.RUST_AVAILABLE = False
py_call = run_python(alloc, cap, buy_call_strategy)
_d.RUST_AVAILABLE = True
rs_call = run_rust(alloc, cap, buy_call_strategy)

print("\n=== BUY CALL ===")
print("Python trade_log:")
print(py_call.trade_log.to_string())
print("\nRust trade_log:")
print(rs_call.trade_log.to_string())
print("Python final:", py_call.balance["total capital"].iloc[-1])
print("Rust final:", rs_call.balance["total capital"].iloc[-1])

# ===================================================
# 5. Tight risk constraints
# ===================================================
from options_portfolio_backtester.portfolio.risk import RiskManager, MaxDelta, MaxVega

rm = RiskManager([MaxDelta(limit=0.01)])
_d.RUST_AVAILABLE = False
py = run_python(alloc, cap, buy_put_strategy, risk_manager=rm)
_d.RUST_AVAILABLE = True
rs = run_rust(alloc, cap, buy_put_strategy, risk_manager=rm)

print("\n=== MAX DELTA TIGHT ===")
print("Python trade_log shape:", py.trade_log.shape)
print("Rust trade_log shape:", rs.trade_log.shape)
print("Python trade_log:")
print(py.trade_log.to_string())
print("Rust trade_log:")
print(rs.trade_log.to_string())

rm2 = RiskManager([MaxVega(limit=50.0)])
_d.RUST_AVAILABLE = False
py2 = run_python(alloc, cap, buy_put_strategy, risk_manager=rm2)
_d.RUST_AVAILABLE = True
rs2 = run_rust(alloc, cap, buy_put_strategy, risk_manager=rm2)

print("\n=== MAX VEGA ===")
print("Python trade_log shape:", py2.trade_log.shape)
print("Rust trade_log shape:", rs2.trade_log.shape)
print("Python trade_log:")
print(py2.trade_log.to_string())
print("Rust trade_log:")
print(rs2.trade_log.to_string())

# ===================================================
# 6. No-match and small capital edge cases
# ===================================================
from options_portfolio_backtester.strategy.strategy import Strategy
from options_portfolio_backtester.strategy.strategy_leg import StrategyLeg
from options_portfolio_backtester.core.types import OptionType as Type, Direction

def tight_strat(schema):
    s = Strategy(schema)
    leg = StrategyLeg("leg_1", schema, option_type=Type.PUT, direction=Direction.BUY)
    leg.entry_filter = (schema.underlying == "SPX") & (schema.dte >= 9999)
    leg.exit_filter = schema.dte <= 30
    s.add_legs([leg])
    return s

_d.RUST_AVAILABLE = False
py = run_python(alloc, cap, tight_strat)
_d.RUST_AVAILABLE = True
rs = run_rust(alloc, cap, tight_strat)

print("\n=== NO MATCH ===")
print("Python final:", py.balance["total capital"].iloc[-1])
print("Rust final:", rs.balance["total capital"].iloc[-1])
print("Python balance head:")
print(py.balance.head().to_string())
print("Rust balance head:")
print(rs.balance.head().to_string())

_d.RUST_AVAILABLE = False
py_small = run_python(alloc, 1000, buy_put_strategy)
_d.RUST_AVAILABLE = True
rs_small = run_rust(alloc, 1000, buy_put_strategy)

print("\n=== SMALL CAPITAL ===")
print("Python trade_log:")
print(py_small.trade_log.to_string())
print("Rust trade_log:")
print(rs_small.trade_log.to_string())
print("Python final:", py_small.balance["total capital"].iloc[-1])
print("Rust final:", rs_small.balance["total capital"].iloc[-1])
