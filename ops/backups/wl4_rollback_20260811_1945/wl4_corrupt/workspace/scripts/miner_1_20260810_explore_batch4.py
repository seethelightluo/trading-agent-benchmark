"""miner_1 batch-4 exploration: macro-beta & conditional-beta factors (2026-08-10).

Tests candidate factors at admission horizon h=10 against the shared gates
(|IC|>=0.007, |ICIR|>=0.084), reports decay, coverage, turnover, and library
correlation vs the 4 base library factors, plus pairwise correlation among new
candidates to avoid persisting a redundant cluster.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr, TRADABLE)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)
H = 10


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var())
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def eval_candidate(panel, name, expected_sign):
    fwd = forward_returns(closes, H)
    ics = rank_ic_series(panel, fwd, 8)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), 8, expected_sign)
    corr, key = max_library_corr(panel, lib)
    m["max_lib_corr"], m["max_corr_key"] = corr, key
    return m


def pct(s):
    s = s.astype(float)
    return s.pct_change()


cands = {}

# --- macro beta drivers (plain 60d) ---
for drv, dname in [("DXY", "dxy"), ("USDJPY", "usdjpy"), ("COPPER", "copper"),
                   ("WTI", "wti"), ("XAU", "xau"), ("BTC", "btc")]:
    if drv not in panels:
        print(f"skip driver {drv}: not in panels")
        continue
    d = pct(panels[drv]["close"])
    panel = rolling_beta(rets, d, 60)
    cands[f"beta_{dname}_60d"] = panel

# --- conditional beta: beta * 20d driver momentum ---
for drv, dname in [("COPPER", "copper"), ("WTI", "wti"), ("XAU", "xau"),
                   ("BTC", "btc"), ("DXY", "dxy")]:
    if drv not in panels:
        continue
    drv_close = panels[drv]["close"].astype(float)
    d = pct(drv_close)
    beta = rolling_beta(rets, d, 60)
    mom20 = drv_close / drv_close.shift(20) - 1.0
    cands[f"cbeta_{dname}_60x20"] = beta.mul(mom20, axis=0)

# --- yield-curve spread beta: beta(asset, d(CN10Y - US10Y), 60) ---
cn10 = panels["CN10Y"]["close"].astype(float)
us10 = panels["US10Y"]["close"].astype(float)
spread = cn10 - us10
cands["beta_yspread_60d"] = rolling_beta(rets, spread.pct_change(), 60)

# --- relative momentum (asset mom minus cross-section mean) at 40d ---
mom40 = closes.shift(5) / closes.shift(45) - 1.0
cands["rel_mom_40d"] = mom40.sub(mom40.mean(axis=1), axis=0)

# --- gold/stock regime tilt: beta(asset, XAU/SPX ratio ret) ---
ratio = panels["XAU"]["close"].astype(float) / panels["SPX"]["close"].astype(float)
cands["beta_goldspx_60d"] = rolling_beta(rets, ratio.pct_change(), 60)

rows = []
for name, panel in cands.items():
    m = eval_candidate(panel, name, 1)
    rows.append((name, m))
    print(f"{name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} covAD={m['coverage_asset_days']:.3f} to={m['turnover_10d_rank']:.2f} "
          f"rho={m['max_lib_corr']:.3f}({m['max_corr_key']}) "
          f"decay10={m['decay']['10']:+.4f} "
          f"{'PASS' if abs(m['ic'])>=0.007 and abs(m['icir'])>=0.084 else ''}")

print("\n--- pairwise cross-sectional rank corr among new candidates ---")
from itertools import combinations
names = [r[0] for r in rows]
for a, b in combinations(names, 2):
    pa, pb = cands[a], cands[b]
    both = pd.concat([pa.stack().rename("a"), pb.stack().rename("b")], axis=1).dropna()
    if len(both) < 30:
        continue
    r = both["a"].corr(both["b"])
    if abs(r) > 0.5:
        print(f"  HIGH {a:22s} vs {b:22s}: rho={r:+.3f} (n={len(both)})")

print("\n--- passers only (sorted by |ICIR|) ---")
passers = sorted([r for r in rows if abs(r[1]['ic']) >= 0.007 and abs(r[1]['icir']) >= 0.084],
                 key=lambda r: -abs(r[1]['icir']))
for name, m in passers:
    print(f"  PASS {name:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} rho={m['max_lib_corr']:.3f}")
