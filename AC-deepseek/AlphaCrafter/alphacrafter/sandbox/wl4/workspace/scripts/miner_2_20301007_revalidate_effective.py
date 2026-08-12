"""miner_2 revalidation (2030-10-07) - revalidate 3 currently effective factors.

Checks |IC|>=0.0070 and |ICIR|>=0.0840 at h=10 on:
  (a) full window 2020..2030-10-04
  (b) recent 2y (last ~500 obs)
  (c) recent 1y (last ~250 obs)
Also computes decay and library correlation vs the 3-factor library.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

panels = load_panels(days=3200)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

# Existing factor panels
lib = {}
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].pct_change(), 60)

H = 10
fwd = forward_returns(closes, H)
fwd_full = fwd

def eval_window(name, panel, fwd, tag, expected_sign):
    ics = rank_ic_series(panel, fwd)
    if len(ics) < 60:
        print(f"{name} [{tag}] SKIP n={len(ics)}", flush=True)
        return
    m = summarize_ic(ics, expected_sign)
    ic, icir = m["ic"], m["icir"]
    gate = (abs(ic) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"{name:26s} [{tag:12s}] IC={ic:+.4f} ICIR={icir:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} GATE={'PASS' if gate else 'FAIL'}", flush=True)
    return m

print("=== revalidation h=10 ===", flush=True)
for name, panel in lib.items():
    exp_sign = 1 if name != "rate_beta_cn10y_60d" else -1
    # full window
    m = eval_window(name, panel, fwd, "full", exp_sign)
    # recent 2y
    sub = fwd.iloc[-520:]
    eval_window(name, panel, sub, "recent2y", exp_sign)
    # recent 1y
    sub1 = fwd.iloc[-260:]
    eval_window(name, panel, sub1, "recent1y", exp_sign)

# library correlations among the 3
print("\n=== pairwise stacked corr among effective factors (full window) ===", flush=True)
for a in lib:
    for b in lib:
        if a < b:
            both = pd.concat([lib[a].stack().rename("a"), lib[b].stack().rename("b")], axis=1).dropna()
            print(f"{a} vs {b}: rho={both['a'].corr(both['b']):+.4f} n={len(both)}", flush=True)

# decay for vol_adj_mom on recent window
print("\n=== decay profiles (full window) ===", flush=True)
for name, panel in lib.items():
    dec = decay_profile(panel, closes, horizons=(1,2,3,5,10,20))
    print(name, dec, flush=True)
