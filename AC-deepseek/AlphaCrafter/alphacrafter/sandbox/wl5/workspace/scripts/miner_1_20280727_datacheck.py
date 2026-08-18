# -*- coding: utf-8 -*-
"""miner_1 2028-07-27: data availability check through visible window."""
import sys
sys.path.insert(0, 'scripts')
import factor_validate as fv
import pandas as pd

VISIBLE = "2028-07-26"
close = fv.closes_panel(visible_through=VISIBLE)
close = close[close.index.dayofweek < 5].copy()
close = close.dropna(how="all", axis=0)
macro = fv.macro_closes(visible_through=VISIBLE)
macro = macro[macro.index.dayofweek < 5].copy()

print(f"CLOSE panel: {close.shape[0]} dates x {close.shape[1]} assets, visible through {VISIBLE}")
print(f"  date range: {close.index.min()} .. {close.index.max()}")
print(f"  dates with >=8 valid: {(close.notna().sum(axis=1) >= 8).sum()}/{len(close)}")
print("\nPer-asset valid counts / last close:")
for c in close.columns:
    last = close[c].dropna()
    print(f"  {c:10s} n={last.shape[0]:5d} last={last.index[-1].date()} close={last.iloc[-1]:.4f}")
print("\nMACRO panel:")
for c in macro.columns:
    last = macro[c].dropna()
    print(f"  {c:10s} n={last.shape[0]:5d} last={last.index[-1].date()} close={last.iloc[-1]:.4f}")

# check the last 30 rows of close dates
print("\nLast 10 dates:", list(close.index[-10:].strftime('%Y-%m-%d')))
# volume availability
df0 = fv.load_panel(["SPX", "BTC", "000300.SH"], "stock", VISIBLE)["SPX"]
print("\nSPX columns:", list(df0.columns))
print("volume non-null:", df0["volume"].notna().mean() if "volume" in df0.columns else "n/a")
