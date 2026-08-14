"""miner_3 probe: verify data columns and current sim date (2035-08-20 cycle)."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from factor_research_lib import TRADABLE, MACRO, load_panels

panels = load_panels(days=5000)
print("panels:", sorted(panels.keys()))
for s in TRADABLE + MACRO:
    if s in panels:
        df = panels[s]
        print(f"{s:10s} rows={len(df):5d} first={df.index[0].date()} last={df.index[-1].date()} "
              f"cols={list(df.columns)} last_close={df['close'].iloc[-1]:.2f}")
