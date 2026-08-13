"""miner_2 probe (2031-04-21) - data availability through visible date."""
import sys
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, TRADABLE, MACRO

panels = load_panels(days=3200)
print("TRADABLE:")
for a in TRADABLE:
    if a in panels:
        print(f"  {a:10s} rows={len(panels[a]):5d} first={panels[a].index[0].date()} last={panels[a].index[-1].date()} last_close={panels[a]['close'].iloc[-1]:.4f}")
    else:
        print(f"  {a:10s} MISSING")
print("MACRO:")
for m in MACRO:
    if m in panels:
        print(f"  {m:10s} rows={len(panels[m]):5d} first={panels[m].index[0].date()} last={panels[m].index[-1].date()}")
    else:
        print(f"  {m:10s} MISSING")
