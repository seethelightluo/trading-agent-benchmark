import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel

t0 = time.time()
panels = load_panels(days=3000)
print(f"load_panels: {time.time()-t0:.1f}s", flush=True)
closes = close_panel(panels)
print(f"closes: {closes.shape} {time.time()-t0:.1f}s", flush=True)
rets = closes.pct_change()
t1 = time.time()
mkt = rets.mean(axis=1)
# quick rank_ic timing
from factor_research_lib import forward_returns, rank_ic_series
fwd = forward_returns(closes, 10)
sig = closes.pct_change().rolling(20).std()
ics = rank_ic_series(sig, fwd, 8)
print(f"rank_ic_series: {time.time()-t1:.1f}s n={len(ics)}", flush=True)
