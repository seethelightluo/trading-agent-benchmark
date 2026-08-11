"""miner_1 batch-4 full validation of passers + pairwise correlation audit (2026-08-10).

Uses the canonical library eval (rank_ic_series etc.) at h=10 for the 4 gate-passers:
beta_dxy_60d, beta_copper_60d, beta_btc_60d, beta_goldspx_60d.
Also recomputes previously persisted beta signals and reports pairwise
cross-sectional correlations to avoid persisting a redundant cluster.
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


def pct(s):
    return s.astype(float).pct_change()


panels_c = {}
panels_c["beta_dxy_60d"] = rolling_beta(rets, pct(panels["DXY"]["close"]), 60)
panels_c["beta_copper_60d"] = rolling_beta(rets, pct(panels["COPPER"]["close"]), 60)
panels_c["beta_btc_60d"] = rolling_beta(rets, pct(panels["BTC"]["close"]), 60)
ratio = panels["XAU"]["close"].astype(float) / panels["SPX"]["close"].astype(float)
panels_c["beta_goldspx_60d"] = rolling_beta(rets, ratio.pct_change(), 60)
# recompute previously persisted beta trio for pairwise audit
panels_c["rate_beta_cn10y_60d"] = rolling_beta(rets, pct(panels["CN10Y"]["close"]), 60)
panels_c["eurusd_beta_60d"] = rolling_beta(rets, pct(panels["EURUSD"]["close"]), 60)
mkt = rets.mean(axis=1)
panels_c["dn_mkt_beta_60d"] = rolling_beta(rets, mkt.where(mkt < 0).fillna(0.0), 60)

results = {}
for name, panel in panels_c.items():
    fwd = forward_returns(closes, H)
    ics = rank_ic_series(panel, fwd, 8)
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"], m["max_corr_factor"] = corr, key
    results[name] = (panel, m)
    status = "PASS" if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 else ""
    print(f"{name:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} covAD={m['coverage_asset_days']:.3f} to={m['turnover_10d_rank']:.2f} "
          f"rho={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']}) "
          f"decay={ {k: round(v,3) for k,v in m['decay_ic_by_horizon'].items()} } {status}")

print("\n--- pairwise cross-sectional correlations among all 7 beta signals ---")
from itertools import combinations
names = list(panels_c.keys())
for a, b in combinations(names, 2):
    pa, pb = panels_c[a], panels_c[b]
    both = pd.concat([pa.stack().rename("a"), pb.stack().rename("b")], axis=1).dropna()
    if len(both) < 30:
        continue
    r = float(both["a"].corr(both["b"]))
    flag = "  <-- HIGH" if abs(r) > 0.5 else ""
    if abs(r) > 0.4:
        print(f"  {a:22s} vs {b:22s}: rho={r:+.3f} n={len(both)}{flag}")

print("\n--- pairwise rank corr among the 4 new passers (CS per-date mean) ---")
new4 = ["beta_dxy_60d", "beta_copper_60d", "beta_btc_60d", "beta_goldspx_60d"]
for a, b in combinations(new4, 2):
    pa, pb = panels_c[a], panels_c[b]
    rhos = []
    for dt in pa.index:
        fa, fb = pa.loc[dt], pb.loc[dt]
        mask = fa.notna() & fb.notna()
        if int(mask.sum()) < 5:
            continue
        x, y = fa[mask].to_numpy(float), fb[mask].to_numpy(float)
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        rhos.append(np.corrcoef(x, y)[0, 1])
    print(f"  {a:22s} vs {b:22s}: mean datewise rho={np.mean(rhos):+.3f} over {len(rhos)} dates")

# save panels for persistence step
import pickle
with open("scripts/_batch4_panels.pkl", "wb") as f:
    pickle.dump({k: v for k, v in panels_c.items()}, f)
print("\nsaved panels to scripts/_batch4_panels.pkl")
