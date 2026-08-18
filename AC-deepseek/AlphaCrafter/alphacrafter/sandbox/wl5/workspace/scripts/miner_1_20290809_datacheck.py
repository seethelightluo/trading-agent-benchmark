"""miner_1 2029-08-09: data availability + volume quality check through 2029-08-08."""
import sys
sys.path.insert(0, "scripts")
from miner_1_20290809_common import (
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
          f"vol_nz={(vol > 0).mean():.2f} vol_mean={vol[vol > 0].mean() if (vol > 0).any() else 0:.3g}")

print("\nMACRO observation signals:")
for s in MACRO:
    df = load_macro(s)
    print(f"  {s:10s} rows={len(df):5d} first={df['date'].iloc[0].date()} last={df['date'].iloc[-1].date()}")

P = ohlcv_panels()
close = P["close"]
print(f"\nClose panel: {close.shape[0]} dates x {close.shape[1]} assets "
      f"({close.index.min().date()}..{close.index.max().date()})")

# volume panel cross-section check: is volume meaningful or constant?
vol = P["volume"]
print("\nVolume panel summary (last 60 rows):")
print(vol.tail(60).describe().loc[["mean", "std", "min", "max"]].T.round(2).to_string())
print("\nVolume non-zero fraction by asset:")
print((vol > 0).mean().round(3).to_string())
print("\nVolume cross-sectional dispersion (cv per date, last 120 rows):")
v = vol.replace(0, np.nan)
cv = v.std(axis=1) / v.mean(axis=1)
print(f"  cv mean={cv.mean():.3f} median={cv.median():.3f}")

# overnight/intraday decomposition feasibility
opn, cls = P["open"], P["close"]
overnight = opn / cls.shift(1) - 1.0
intraday = cls / opn - 1.0
print("\nOvernight vs intraday return stats:")
print(f"  overnight: mean={overnight.mean():.6f} std={overnight.std():.6f} nz={(overnight.abs() > 1e-12).mean():.3f}")
print(f"  intraday:  mean={intraday.mean():.6f} std={intraday.std():.6f} nz={(intraday.abs() > 1e-12).mean():.3f}")
print("\nDate range of usable data confirmed OK.")
