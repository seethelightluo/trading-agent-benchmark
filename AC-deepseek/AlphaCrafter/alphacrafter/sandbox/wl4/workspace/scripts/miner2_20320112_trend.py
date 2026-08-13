"""miner2_20320112: refine trend_strength family + check library correlation + decay.
Candidate: trend_strength (close - sma_k)/std_k for k in {20,40,60,90,120},
plus vol-scaled variant and z-scored cross-sectional variant.
Also checks max abs correlation vs existing effective library artifacts.
"""
import sys, json
sys.path.insert(0, "scripts")
from miner2_20320112_validator import load_panel, forward_returns, full_metrics, max_library_corr, make_signal_artifact
import numpy as np
import pandas as pd

px, mx = load_panel()
ret = px.pct_change()
fwd = forward_returns(px)

def ic_summary(factor_df, label):
    m = full_metrics(factor_df, fwd, min_valid=8)
    h10 = m["horizons"]["10"]
    h5 = m["horizons"]["5"]
    print(f"{label:32s} h5 ic={h5['ic']:+.4f} icir={h5['icir']:+.3f} | h10 ic={h10['ic']:+.4f} icir={h10['icir']:+.3f} hit={h10['hit']:.2f} n={h10['n']:4d} cov_ad={m['coverage_asset_days']:.3f} cov_d8={m['coverage_dates_ge8']:.3f} to={m['turnover_10d_rank']:.2f}")
    return m

for k in [20, 40, 60, 90, 120]:
    sma = px.rolling(k).mean()
    std = px.rolling(k).std()
    f = (px - sma) / std
    ic_summary(f, f"trend_strength_{k}d")

# cross-sectional z-scored version
for k in [60]:
    sma = px.rolling(k).mean()
    std = px.rolling(k).std()
    raw = (px - sma) / std
    z = raw.sub(raw.median(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    ic_summary(z, f"trend_z_{k}d")

# vol-scaled (per-asset z of deviation)
for k in [60]:
    sma = px.rolling(k).mean()
    dev = px - sma
    z_asset = dev.div(dev.rolling(20).std(), axis=0)
    ic_summary(z_asset, f"trend_devvol_{k}d")

# best candidate full metrics + library corr
f_best = (px - px.rolling(60).mean()) / px.rolling(60).std()
m = full_metrics(f_best, fwd, min_valid=8)
print("\nBEST full:", json.dumps(m, indent=1))
lc = max_library_corr(f_best)
print("max_library_corr:", lc)
