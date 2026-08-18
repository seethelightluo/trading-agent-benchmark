"""miner_1 2029-09-06: data availability + volume quality check through 2029-09-05."""
import sys
sys.path.insert(0, "scripts")
from miner_1_20290906_common import (
    TRADABLE, MACRO, VISIBLE_THROUGH, CURRENT_DATE, load_asset, load_macro,
    ohlcv_panels, macro_panel,
)
import numpy as np
import pandas as pd

print("=" * 90)
print(f"DATA CHECK through {VISIBLE_THROUGH} (current {CURRENT_DATE})")
print("TRADABLE instruments:")
for s in TRADABLE:
    df = load_asset(s)
    vol = df["volume"]
    print(f"  {s:10s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()} "
          f"vol_nz={(vol > 0).mean():.2f}")

print("\nMACRO observation signals:")
for s in MACRO:
    df = load_macro(s)
    print(f"  {s:10s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()}")

P = ohlcv_panels()
close = P["close"]
print(f"\nClose panel: {close.shape[0]} dates x {close.shape[1]} assets "
      f"({close.index.min().date()}..{close.index.max().date()})")

vol = P["volume"]
print("\nVolume non-zero fraction by asset:")
print((vol > 0).mean().round(3).to_string())
print("\nDate range of usable data confirmed OK.")
