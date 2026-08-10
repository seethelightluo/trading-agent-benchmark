"""miner_1 2026-07-30 cycle 27 (part 2): redundancy & stability screen for passers.

Decides which of {er_20d, mkt_beta_60d, er_ratio_20x60} to persist:
- pairwise rank rho among passers (avoid persisting near-duplicates)
- last-2yr vs first-2yr IC consistency (regime stability screen, per btc_beta precedent)
- turnover detail
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, panel_rank_corr, turnover_rank)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---- rebuild the 3 passers ----
def er_factory(n):
    def f(s):
        diff = s.diff().abs().rolling(n).sum()
        net = (s - s.shift(n)).abs()
        return net / diff
    return f

er20 = per_asset(panel, er_factory(20))
er60 = per_asset(panel, er_factory(60))
er_ratio = er20 - er60

rets = panel.pct_change()
ew_ret = rets.mean(axis=1, skipna=True)
def mkt_beta_factory(window=60, minp=30):
    def f(s):
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), ew_ret.reindex(ar.index).rename("m")], axis=1).dropna()
        cov = df["a"].rolling(window, min_periods=minp).cov(df["m"])
        var = df["m"].rolling(window, min_periods=minp).var()
        return (cov / var).reindex(s.index)
    return f
mkt_beta = per_asset(panel, mkt_beta_factory(60))

signals = {"er_20d": er20, "er_ratio_20x60": er_ratio, "mkt_beta_60d": mkt_beta}

print("=== pairwise rank rho among passers (full sample) ===")
names = list(signals.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        r = panel_rank_corr(signals[names[i]], signals[names[j]])
        print(f"  {names[i]:16s} vs {names[j]:16s} = {r:+.4f}")

print("\n=== turnover_10d_rank ===")
for fid, sig in signals.items():
    print(f"  {fid}: {turnover_rank(sig, step=ADM_H):.3f}")

print("\n=== IC stability: first-half vs last-half of sample ===")
for fid, sig in signals.items():
    half = panel.index[len(panel.index)//2]
    ic1 = compute_ic(sig.loc[panel.index < half], fwd_cache[str(ADM_H)].loc[panel.index < half]).dropna()
    ic2 = compute_ic(sig.loc[panel.index >= half], fwd_cache[str(ADM_H)].loc[panel.index >= half]).dropna()
    print(f"  {fid:16s} early: {ic1.mean():+.4f}/{ic1.mean()/ic1.std():+.3f}/n={len(ic1)}  "
          f"late: {ic2.mean():+.4f}/{ic2.mean()/ic2.std():+.3f}/n={len(ic2)}")

print("\n=== last 250d IC (freshness) ===")
for fid, sig in signals.items():
    sub = sig.index[-250:]
    ic = compute_ic(sig.loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
    if len(ic) >= 30:
        print(f"  {fid:16s} last250d: {ic.mean():+.4f}/{ic.mean()/ic.std():+.3f}/n={len(ic)}")
    else:
        print(f"  {fid:16s} last250d: insufficient dates n={len(ic)}")
