"""miner_2 2026-09-14: probe current data state - dates, availability, last completed day."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

panels = load_panels(days=3000)
closes = close_panel(panels)

print("tradable assets loaded:", list(panels.keys() & set(TRADABLE)))
print("macro assets loaded:", list(panels.keys() & set(MACRO)))
print()
for a in TRADABLE:
    df = panels.get(a)
    if df is None:
        print(f"{a:10s} MISSING")
        continue
    print(f"{a:10s} rows={len(df):5d} first={df.index[0].date()} last={df.index[-1].date()}")
print()
print("union calendar: first =", closes.index[0].date(), " last =", closes.index[-1].date(), " n =", len(closes.index))
print("last 5 union dates:", [d.date() for d in closes.index[-5:]])
# macro last dates
for a in MACRO:
    df = panels.get(a)
    if df is not None:
        print(f"macro {a:8s} last={df.index[-1].date()} rows={len(df)}")
