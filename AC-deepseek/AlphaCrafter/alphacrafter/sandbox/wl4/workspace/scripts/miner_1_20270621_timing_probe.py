"""miner_1 timing probe - identify slow stage in batchK screen."""
import sys, time, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, max_library_corr, TRADABLE)

os.makedirs("logs", exist_ok=True)
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

log("start")
panels = load_panels(days=3000)
log(f"load_panels done {len(panels)} symbols")
closes = close_panel(panels)
rets = closes.pct_change()
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
log(f"panels built closes {closes.shape} {closes.index.min().date()}..{closes.index.max().date()}")

# time one rank_ic call
fwd = forward_returns(closes, 10)
t1 = time.time()
ics = rank_ic_series(closes.pct_change().rolling(20).std(), fwd, 8)
log(f"rank_ic_series one call: {time.time()-t1:.1f}s, n={len(ics)}")

# time max_library_corr one call
t1 = time.time()
cand = (closes - closes.shift(59)).abs() / rets.abs().rolling(60).sum().replace(0, np.nan)
lib = {"mom": closes.shift(5) / closes.shift(15) - 1.0}
corr, key = max_library_corr(cand, lib)
log(f"max_library_corr one call: {time.time()-t1:.1f}s -> {corr} {key}")

# time autocorr vectorized
t1 = time.time()
def autocorr_lag_vec(x, lag=5, win=60):
    y = x.shift(lag)
    mx = x.rolling(win).mean()
    my = y.rolling(win).mean()
    cov = (x * y).rolling(win).mean() - mx * my
    vx = x.rolling(win).var()
    vy = y.rolling(win).var()
    return cov / np.sqrt(vx * vy).replace(0, np.nan)
_ = pd.DataFrame({a: autocorr_lag_vec(rets[a], 5, 60) for a in closes.columns}, index=rets.index)
log(f"autocorr all assets: {time.time()-t1:.1f}s")

# time vol_price_lead (rolling corr with shifted vol)
t1 = time.time()
_ = pd.DataFrame({a: rets[a].rolling(20).corr(vol_panel[a].shift(1)) for a in closes.columns}, index=rets.index)
log(f"vol_price_lead all assets: {time.time()-t1:.1f}s")

# rolling skew
t1 = time.time()
_ = rets.rolling(20).skew()
log(f"rolling skew: {time.time()-t1:.1f}s")

log("probe done")
