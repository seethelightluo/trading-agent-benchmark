"""miner_1 batch-4b: date-wise pairwise correlation audit for final 3 candidates.

Candidates: beta_goldspx_40d, cbeta_cn10y_60x20, mom20_x_vixz.
Check date-wise mean cross-sectional correlation vs ALL persisted library
signals (4 base + rate_beta_cn10y_60d + eurusd_beta_60d + dn_mkt_beta_60d)
and among themselves, then run canonical full eval for each.
"""
import sys, pickle
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var())
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def pct(s):
    return s.astype(float).pct_change()


# --- candidate panels ---
gs = panels["XAU"]["close"].astype(float) / panels["SPX"]["close"].astype(float)
cand = {}
cand["beta_goldspx_40d"] = rolling_beta(rets, gs.pct_change(), 40)
cn10 = panels["CN10Y"]["close"].astype(float)
b_cn = rolling_beta(rets, pct(cn10), 60)
cand["cbeta_cn10y_60x20"] = b_cn.mul(np.sign(cn10 / cn10.shift(20) - 1.0), axis=0)
vix = panels["VIX"]["close"].astype(float)
vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
mom20 = closes.shift(5) / closes.shift(25) - 1.0
cand["mom20_x_vixz"] = mom20.mul(vix_z, axis=0)

# --- persisted library panels (recomputed) ---
persisted = {}
persisted["rate_beta_cn10y_60d"] = rolling_beta(rets, pct(panels["CN10Y"]["close"]), 60)
persisted["eurusd_beta_60d"] = rolling_beta(rets, pct(panels["EURUSD"]["close"]), 60)
mkt = rets.mean(axis=1)
persisted["dn_mkt_beta_60d"] = rolling_beta(rets, mkt.where(mkt < 0).fillna(0.0), 60)
lib_all = dict(lib)
lib_all.update(persisted)

print("--- date-wise mean cross-sectional rho: candidate vs each library signal ---")
for name, panel in cand.items():
    for lname, lpanel in lib_all.items():
        rhos = []
        for dt in panel.index:
            fa, fb = panel.loc[dt], lpanel.loc[dt]
            mask = fa.notna() & fb.notna()
            if int(mask.sum()) < 5:
                continue
            x, y = fa[mask].to_numpy(float), fb[mask].to_numpy(float)
            if x.std() < 1e-12 or y.std() < 1e-12:
                continue
            rhos.append(np.corrcoef(x, y)[0, 1])
        r = np.mean(rhos) if rhos else np.nan
        flag = "  <-- HIGH" if abs(r) > 0.5 else ""
        if abs(r) > 0.3:
            print(f"  {name:20s} vs {lname:20s}: datewise rho={r:+.3f} n={len(rhos)}{flag}")

print("--- candidate vs candidate datewise ---")
cn = list(cand.keys())
for i in range(len(cn)):
    for j in range(i + 1, len(cn)):
        a, b = cn[i], cn[j]
        rhos = []
        for dt in cand[a].index:
            fa, fb = cand[a].loc[dt], cand[b].loc[dt]
            mask = fa.notna() & fb.notna()
            if int(mask.sum()) < 5:
                continue
            x, y = fa[mask].to_numpy(float), fb[mask].to_numpy(float)
            if x.std() < 1e-12 or y.std() < 1e-12:
                continue
            rhos.append(np.corrcoef(x, y)[0, 1])
        print(f"  {a} vs {b}: datewise rho={np.mean(rhos):+.3f} n={len(rhos)}")

print("\n--- canonical full eval at h=10 ---")
res = {}
for name, panel in cand.items():
    fwd = forward_returns(closes, 10)
    ics = rank_ic_series(panel, fwd, 8)
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), 8, 1)
    corr, key = max_library_corr(panel, lib_all)
    m["max_abs_library_correlation"], m["max_corr_factor"] = corr, key
    res[name] = (panel, m)
    print(f"{name:20s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} covAD={m['coverage_asset_days']:.3f} to={m['turnover_10d_rank']:.2f} "
          f"rhoAll={m['max_abs_library_correlation']:.3f}({m['max_corr_factor']}) "
          f"decay10={m['decay_ic_by_horizon']['10']:+.4f}")

with open("scripts/_batch4b_final.pkl", "wb") as f:
    pickle.dump({k: v for k, v in res.items()}, f)
print("saved scripts/_batch4b_final.pkl")
