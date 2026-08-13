"""Quick timing probe for miner_2 batch AA."""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel, forward_returns, rank_ic_series

t0 = time.time()
panels = load_panels(days=4000)
print(f"load_panels(4000): {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
closes = close_panel(panels)
rets = closes.pct_change()
print(f"close_panel: {time.time()-t0:.1f}s | shape {closes.shape}", flush=True)
t0 = time.time()
fwd = forward_returns(closes, 10)
print(f"forward_returns: {time.time()-t0:.1f}s", flush=True)

# vectorized rolling beta test
vix = panels["VIX"]["close"].astype(float)
vix = vix.reindex(closes.index).ffill()
t0 = time.time()
xr = vix.pct_change()
xy = rets.mul(xr, axis=0)
x2 = xr.pow(2)
my = rets.rolling(60).mean()
mx = xr.rolling(60).mean()
mxy = xy.rolling(60).mean()
mx2 = x2.rolling(60).mean()
cov = mxy - my.mul(mx, axis=0)
var = mx2 - mx.pow(2)
beta = cov.div(var.replace(0, np.nan), axis=0)
print(f"vectorized beta: {time.time()-t0:.1f}s | beta shape {beta.shape} nan% {beta.isna().mean().mean():.2f}", flush=True)

# rank_ic timing for one candidate
t0 = time.time()
ics = rank_ic_series(beta, fwd)
print(f"rank_ic_series: {time.time()-t0:.1f}s | n={len(ics)}", flush=True)
