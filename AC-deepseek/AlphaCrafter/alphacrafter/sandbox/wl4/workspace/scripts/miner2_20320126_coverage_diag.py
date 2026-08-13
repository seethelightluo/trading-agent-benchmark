"""miner2_20320126: diagnose coverage for trend_strength family.
Why is coverage_asset_days ~0.40 while rate_mom has 0.74?
Check per-asset valid counts and per-year valid-date counts."""
import sys, json
sys.path.insert(0, "scripts")
from miner2_20320112_validator import load_panel
import numpy as np
import pandas as pd

px, mx = load_panel()
print(f"Panel shape: {px.shape} (dates x assets)")
print(f"Date range: {px.index.min()} .. {px.index.max()}")

# per-asset non-null counts and date ranges
print("\nPer-asset raw close coverage:")
for c in px.columns:
    s = px[c].dropna()
    print(f"  {c:10s} n={len(s):5d} from {s.index.min().date()} to {s.index.max().date()}")

# how many dates have >=8 assets with close data
ge8 = (px.notna().sum(axis=1) >= 8)
print(f"\nDates with >=8 closes: {ge8.sum()} / {len(px)} ({ge8.mean():.3f})")
# weekday vs weekend split
wd = px.index.dayofweek < 5
print(f"  weekday dates: {(ge8 & wd).sum()} / {wd.sum()} ({((ge8 & wd).sum()/wd.sum()):.3f})")
print(f"  weekend dates: {(ge8 & ~wd).sum()} / {(~wd).sum()} ({((ge8 & ~wd).sum()/(~wd).sum()):.3f})")

# per-year coverage for 60d rolling std (requires 60 prior days + data)
std60 = px.rolling(60).std()
f_valid = std60.notna()
print("\nPer-year factor-valid date counts (>=8 assets):")
yr = f_valid.sum(axis=1) >= 8
for y in range(2020, 2033):
    m = (px.index.year == y)
    print(f"  {y}: dates={m.sum():5d} valid_ge8={yr[m].sum():5d} ({yr[m].mean():.3f})")

# per-asset valid days of the 60d factor
print("\nPer-asset valid days for 60d std factor:")
for c in px.columns:
    print(f"  {c:10s} n={int(f_valid[c].sum()):5d} frac={f_valid[c].mean():.3f}")
