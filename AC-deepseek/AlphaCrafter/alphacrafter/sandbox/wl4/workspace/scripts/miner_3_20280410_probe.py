"""miner_3 2028-04-10 probe: confirm data visibility window for the 15-asset universe."""
import sys, time
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, TRADABLE

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
print(f"load {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}")
print("last completed trading day:", closes.index.max().date())
print("\nper-asset last close date:")
for a in TRADABLE:
    s = closes[a].dropna()
    if len(s):
        print(f"  {a:10s} last={s.index[-1].date()} n={len(s)} last_close={s.iloc[-1]:.4f}")
    else:
        print(f"  {a:10s} NO DATA")
print("\nper-asset recent non-null count (last 60 rows of close panel):")
print(closes.tail(60).notna().sum())
