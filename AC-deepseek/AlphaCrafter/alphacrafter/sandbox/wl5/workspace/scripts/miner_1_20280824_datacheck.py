# -*- coding: utf-8 -*-
"""miner_1 2028-08-24: data availability check through visible window (2028-08-23)."""
import sys
sys.path.insert(0, 'scripts')
import factor_validate as fv
import pandas as pd

VISIBLE = "2028-08-23"
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

print("\nLast 10 dates:", list(close.index[-10:].strftime('%Y-%m-%d')))
