"""miner_1 probe: check data availability as of current sim date 2035-08-20."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from factor_research_lib import TRADABLE, MACRO, load_panels

panels = load_panels(days=5000)
print("panels loaded:", sorted(panels.keys()))
for s in TRADABLE + MACRO:
    if s in panels:
        df = panels[s]
        print(f"{s:10s} rows={len(df):5d} first={df.index[0].date()} last={df.index[-1].date()} last_close={df['close'].iloc[-1]:.2f}")
    else:
        print(f"{s:10s} MISSING")
