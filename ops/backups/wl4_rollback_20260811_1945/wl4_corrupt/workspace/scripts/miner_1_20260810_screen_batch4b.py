"""miner_1 batch-4b screening: orthogonal variants of passers + new conditional ideas.

Goal: find factors that pass IC/ICIR gates AND have low pairwise correlation
(<0.5) with already-persisted library signals (rate_beta_cn10y_60d,
eurusd_beta_60d, dn_mkt_beta_60d + 4 base factors).
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 library_signals, max_library_corr)

panels = load_panels()
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)
H = 10
fwd10 = forward_returns(closes, H)

# recompute previously persisted beta signals for correlation audit
def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = (z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var())
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def pct(s):
    return s.astype(float).pct_change()


persisted = {}
persisted["rate_beta_cn10y_60d"] = rolling_beta(rets, pct(panels["CN10Y"]["close"]), 60)
persisted["eurusd_beta_60d"] = rolling_beta(rets, pct(panels["EURUSD"]["close"]), 60)
mkt = rets.mean(axis=1)
persisted["dn_mkt_beta_60d"] = rolling_beta(rets, mkt.where(mkt < 0).fillna(0.0), 60)
lib_all = dict(lib)
lib_all.update(persisted)


def fast_rank_ic(panel, fwd, min_valid=8):
    f_rank = panel.rank(axis=1)
    r_rank = fwd.rank(axis=1)
    dates, ics = [], []
    for dt in panel.index:
        if dt not in fwd.index:
            continue
        fr, rr = f_rank.loc[dt], r_rank.loc[dt]
        mask = fr.notna() & rr.notna()
        if int(mask.sum()) < min_valid:
            continue
        fv, rv = fr[mask].to_numpy(float), rr[mask].to_numpy(float)
        if fv.std() < 1e-12 or rv.std() < 1e-12:
            continue
        ic = float(np.corrcoef(fv, rv)[0, 1])
        if not np.isnan(ic):
            dates.append(dt)
            ics.append(ic)
    return np.array(ics)


cands = {}
# goldspx beta window variants
gs = panels["XAU"]["close"].astype(float) / panels["SPX"]["close"].astype(float)
for w in (40, 90, 120):
    cands[f"beta_goldspx_{w}d"] = rolling_beta(rets, gs.pct_change(), w)
# china divergence beta: beta(asset, 000300/SPX ratio)
cspx = panels["000300.SH"]["close"].astype(float) / panels["SPX"]["close"].astype(float)
cands["beta_cnspx_60d"] = rolling_beta(rets, cspx.pct_change(), 60)
# conditional US rate beta: beta(asset, US10Y chg) * sign(US10Y 20d chg)
us10 = panels["US10Y"]["close"].astype(float)
b_us = rolling_beta(rets, pct(us10), 60)
cands["cbeta_us10y_60x20"] = b_us.mul(np.sign(us10 / us10.shift(20) - 1.0), axis=0)
# conditional CN rate beta: beta(asset, CN10Y chg) * sign(CN10Y 20d chg)
cn10 = panels["CN10Y"]["close"].astype(float)
b_cn = rolling_beta(rets, pct(cn10), 60)
cands["cbeta_cn10y_60x20"] = b_cn.mul(np.sign(cn10 / cn10.shift(20) - 1.0), axis=0)
# VIX-regime conditional momentum: mom20d * VIX z-score (60d)
vix = panels["VIX"]["close"].astype(float)
vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
mom20 = closes.shift(5) / closes.shift(25) - 1.0
cands["mom20_x_vixz"] = mom20.mul(vix_z, axis=0)
# VIX-regime conditional downside beta: dn_mkt_beta * VIX z-score
cands["dnbeta_x_vixz"] = persisted["dn_mkt_beta_60d"].mul(vix_z, axis=0)
# down/up vol ratio 40d
r = rets
up_vol = r.where(r > 0).rolling(40).std()
dn_vol = r.where(r < 0).rolling(40).std()
cands["down_up_vol40"] = dn_vol / up_vol

print(f"built {len(cands)} candidates")
rows = []
for name, panel in cands.items():
    ic = fast_rank_ic(panel, fwd10, 8)
    icv = ic.mean()
    icir = icv / ic.std(ddof=1) if ic.std(ddof=1) > 0 else 0.0
    corr, key = max_library_corr(panel, lib_all)
    status = "PASS" if abs(icv) >= 0.007 and abs(icir) >= 0.084 else ""
    rows.append((name, icv, icir, len(ic), corr, key, status))
    print(f"{name:22s} IC={icv:+.4f} ICIR={icir:+.4f} n={len(ic):4d} "
          f"rhoAll={corr:.3f}({key}) {status}")

print("\n--- passers ---")
for name, icv, icir, n, corr, key, st in rows:
    if st:
        print(f"  PASS {name:22s} IC={icv:+.4f} ICIR={icir:+.4f} rhoAll={corr:.3f}({key})")
