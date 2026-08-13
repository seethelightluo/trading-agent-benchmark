import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape {closes.shape}", flush=True)
print("last date:", closes.index.max(), "first date:", closes.index.min())
print("\nlast date per asset:")
print(closes.apply(lambda s: s.dropna().index.max()).to_string(), flush=True)
print("\nNaN count per asset (last 400 rows):")
print(closes.tail(400).isna().sum().to_string(), flush=True)

# valid count distribution
valid = closes.notna()
nvalid = valid.sum(axis=1)
print(f"\nvalid-count dist: min={nvalid.min()} p10={nvalid.quantile(0.1):.0f} med={nvalid.median():.0f} max={nvalid.max()}")
print(f"dates with >=8 valid: {(nvalid>=8).sum()} / {len(nvalid)}")

# macro panels
for m in ["VIX","DXY","USDCNY","USDJPY","EURUSD"]:
    if m in panels:
        print(f"{m}: last={panels[m].index.max().date()} rows={len(panels[m])}")
print(f"total time {time.time()-t0:.1f}s")
