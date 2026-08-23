"""Probe data availability as of current sim date 2029-09-06."""
from alphacrafter.sim.utils import (
    get_stock_daily_data,
    get_index_daily_data,
    get_account_dict,
)

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

print("=== get_stock_daily_data (tradables) ===")
for sym in WATCH:
    df = get_stock_daily_data(symbol=sym, days=5000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} cols={list(df.columns)}")
    else:
        print(f"{sym:10s} NONE")

print("=== get_index_daily_data (tradables) ===")
for sym in WATCH:
    df = get_index_daily_data(symbol=sym, days=5000)
    if df is not None and len(df):
        print(f"{sym:10s} len={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    else:
        print(f"{sym:10s} NONE")

print("=== account ===")
acc = get_account_dict()
print({k: acc[k] for k in ['total_assets', 'net_assets', 'available_cash', 'watch_list'] if k in acc})
print("n positions:", len(acc.get('positions', [])))