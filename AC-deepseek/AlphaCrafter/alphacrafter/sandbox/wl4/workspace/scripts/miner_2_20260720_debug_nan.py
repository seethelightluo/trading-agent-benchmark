"""Debug: why close-based candidate factors have ~0.13 coverage (2/15 assets)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from factor_research_lib import load_panels, close_panel, TRADABLE

panels = load_panels(3000)
closes = close_panel(panels)
print("closes shape:", closes.shape)
print("closes columns:", list(closes.columns))
print("closes index range:", closes.index.min(), "->", closes.index.max())

WINDOW = (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-07-15"))
closes_w = closes.loc[(closes.index >= WINDOW[0]) & (closes.index <= WINDOW[1])]
print("windowed closes shape:", closes_w.shape)

rets = closes_w.pct_change()
vol20 = rets.rolling(20).std()
print("\nvol20 notna counts per asset:")
print(vol20.notna().sum())

mom60 = closes_w.shift(5) / closes_w.shift(65) - 1.0
print("\nmom60 notna counts per asset:")
print(mom60.notna().sum())

ram = mom60 / vol20
print("\nrisk_adj_mom notna counts per asset:")
print(ram.notna().sum())

# check individual panel coverage for one asset
for a in TRADABLE[:3]:
    df = panels[a]
    print(f"\n{a}: rows={len(df)} cols={list(df.columns)}")
    print("  close last 3:", df["close"].dropna().tail(3).values)
    print("  high last 3:", df["high"].dropna().tail(3).values)
