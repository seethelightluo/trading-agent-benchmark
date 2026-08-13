"""miner_3 probe 2033-11-14 - verify data availability and panel shapes."""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

panels = load_panels(days=5000)
print("tradable loaded:", {k: len(v) for k, v in panels.items() if k in TRADABLE})
print("macro loaded:", {k: len(v) for k, v in panels.items() if k in MACRO})
closes = close_panel(panels)
print("closes shape:", closes.shape)
print("closes index:", closes.index.min().date(), "..", closes.index.max().date())
print("last 3 rows dates:", list(closes.index[-3:].strftime('%Y-%m-%d')))
for m in MACRO:
    if m in panels:
        print(m, "last date:", panels[m].index.max().date(), "rows:", len(panels[m]))
# check missing macro in close panel
for m in MACRO:
    if m not in panels:
        print("MISSING macro:", m)
