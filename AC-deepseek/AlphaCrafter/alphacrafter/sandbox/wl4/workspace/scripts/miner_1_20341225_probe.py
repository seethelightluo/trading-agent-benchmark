"""miner_1 2034-12-25 - probe current data state before factor exploration."""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, ret_panel, TRADABLE, MACRO

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = ret_panel(panels)
print("closes:", closes.shape, closes.index.min().date(), "..", closes.index.max().date())
print("assets:", list(closes.columns))
valid = closes.notna()
print("dates with >=8 valid:", int((valid.sum(axis=1) >= 8).sum()), "of", len(closes))
print("per-asset valid days:")
print(valid.sum().to_string())
# macro availability
for m in MACRO:
    if m in panels:
        print(m, "rows:", len(panels[m]), panels[m].index.min().date(), "..", panels[m].index.max().date())
    else:
        print(m, "MISSING")
# last 5 closes
print("last 5 closes:")
print(closes.tail(5).round(2).to_string())
