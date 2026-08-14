"""miner_2 2034-10-02: probe data endpoint & truncation behavior.
Current simulated date is 2034-10-02. Factor research must only use data
through the previous completed trading day (no lookahead)."""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, TRADABLE

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = closes.pct_change()

print("n_tradable_panels:", len([a for a in TRADABLE if a in panels]))
print("n_macro_panels:", len([a for a in ["VIX","DXY","USDCNY","USDJPY","EURUSD"] if a in panels]))
print("panel max date:", closes.index.max())
print("panel min date:", closes.index.min())
print("n_dates:", len(closes), "| n_assets:", closes.shape[1])
print("last 5 dates:", [d.strftime("%Y-%m-%d") for d in closes.index[-5:]])
print("first 3 dates:", [d.strftime("%Y-%m-%d") for d in closes.index[:3]])

# check per-asset last date (any asset with different endpoint = truncation issue)
for a in closes.columns:
    last = closes[a].dropna().index.max()
    print(f"  {a:12s} last_valid={last.strftime('%Y-%m-%d')} n={int(closes[a].notna().sum())}")

# confirm there is no data on/after 2034-10-02 (the current date)
print("any row >= 2034-10-02 in closes:", bool((closes.index >= "2034-10-02").any()))
# last available date for macro too
for s in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    if s in panels:
        print(f"macro {s:8s} last={panels[s].index.max()}")

# quick sanity on returns availability
print("rets tail 3 rows:")
print(rets.tail(3).round(4))
print("DONE")
