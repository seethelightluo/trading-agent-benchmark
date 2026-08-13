"""Data coverage audit: when does each asset's price series start/end, and how many
assets are valid per year? This informs re-validation conclusions (e.g., CN10Y gaps)."""
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner2_20330919_common import load_price_panel, TRADABLE, load_asset

px = load_price_panel()
print("Per-asset coverage:")
for s in TRADABLE:
    col = px[s]
    first = col.first_valid_index()
    last = col.last_valid_index()
    n = col.notna().sum()
    print(f"  {s:12s} first={first.date() if first is not None else None} "
          f"last={last.date() if last is not None else None} n_valid={n}")

print("\nValid-asset count per year (>=8 required for IC):")
pxy = px.copy()
pxy['year'] = px.index.year
cnt = pxy.groupby('year').apply(lambda d: (d.drop(columns='year').notna().sum(axis=1) >= 8).mean())
print(cnt.round(3).to_string())

# CN10Y specifics
cn = load_asset("CN10Y")
print("\nCN10Y tail:")
print(cn.tail(6).to_string())
cn2 = load_asset("CN10Y")
print("CN10Y last date:", cn2['date'].max())
