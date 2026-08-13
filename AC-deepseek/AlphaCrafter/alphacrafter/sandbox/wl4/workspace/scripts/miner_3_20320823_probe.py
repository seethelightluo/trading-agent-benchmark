"""miner_3 probe (2032-08-23): check data availability through last completed day."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE

panels = load_panels(days=4000)
closes = close_panel(panels)
print("closes shape:", closes.shape)
print("date range:", closes.index.min().date(), "->", closes.index.max().date())
print("\nper-asset latest date & n rows:")
for a in TRADABLE:
    if a in panels:
        print(f"  {a:12s} n={len(panels[a]):5d} last={panels[a].index.max().date()} last_close={panels[a]['close'].iloc[-1]:.4f}")
    else:
        print(f"  {a:12s} MISSING")
print("\nmacro panels:", list(panels.keys() - set(TRADABLE)))
