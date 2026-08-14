"""miner_1 (2034-07-24): probe data availability and current library state."""
import sys, json
sys.path.insert(0, "scripts")
import pandas as pd
from factor_research_lib import load_panels, close_panel

panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
print("panel assets:", sorted(panels.keys()))
print("closes shape:", closes.shape)
print("last date:", closes.index.max().date())
print("first date:", closes.index.min().date())
print("last 5 dates:", [d.date() for d in closes.index[-5:]])
print("assets with last close notna:", closes.iloc[-1].notna().sum(), "/", closes.shape[1])
print("min rows:", closes.notna().sum().min(), "max rows:", closes.notna().sum().max())

# current library status
meta = {}
import pathlib
for p in sorted(pathlib.Path("factors").glob("*.json")):
    try:
        d = json.loads(p.read_text())
        meta[d["factor_id"]] = d.get("validation", {}).get("status")
    except Exception as e:
        print("bad json", p, e)
print("library factors:", meta)
