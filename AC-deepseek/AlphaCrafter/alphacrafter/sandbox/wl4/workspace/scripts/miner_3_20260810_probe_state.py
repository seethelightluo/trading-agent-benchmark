"""Probe current data state for miner_3: latest dates, coverage, current date."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

panels = load_panels(days=3000)
print("Panels loaded:", {k: len(v) for k, v in panels.items()})

closes = close_panel(panels)
print("\nClose panel shape:", closes.shape)
print("Close panel index range:", closes.index[0].date(), "->", closes.index[-1].date())

# per-asset last date + row counts
print("\nPer-asset last date / n_days:")
for a in TRADABLE + MACRO:
    if a in panels:
        df = panels[a]
        print(f"  {a:10s} last={df.index[-1].date()} n={len(df):4d}")

# min valid count per date across tradable
valid = closes.notna()
ge8 = (valid.sum(axis=1) >= 8)
print("\nDates with >=8 valid tradable instruments:", int(ge8.sum()), "/", len(closes))

# volume availability sample
print("\nVolume non-null share per asset (tradable):")
for a in TRADABLE:
    if a in panels and "volume" in panels[a]:
        s = panels[a]["volume"]
        print(f"  {a:10s} {float(s.notna().mean()):.3f} nonzero_share={float((s.fillna(0) > 0).mean()):.3f}")
    else:
        print(f"  {a:10s} NO volume column")
