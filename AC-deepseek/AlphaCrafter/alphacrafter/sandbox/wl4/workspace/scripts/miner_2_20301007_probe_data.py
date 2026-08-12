"""miner_2 probe (2030-10-07) - check data availability & last dates for all panels."""
import sys
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, TRADABLE, MACRO

panels = load_panels(days=3200)
print("assets loaded:", sorted(panels.keys()))
for a in TRADABLE + MACRO:
    df = panels.get(a)
    if df is None:
        print(f"{a:10s} MISSING")
    else:
        print(f"{a:10s} rows={len(df):5d} {df.index.min().date()} .. {df.index.max().date()}")
closes = close_panel(panels)
print("\nclose panel shape:", closes.shape)
print("last row:\n", closes.iloc[-1])
print("n_assets with last-date close:", closes.iloc[-1].notna().sum())
# how many dates in last ~6 months
print("\nlast 130 rows date range:", closes.index[-130].date(), "..", closes.index[-1].date())
