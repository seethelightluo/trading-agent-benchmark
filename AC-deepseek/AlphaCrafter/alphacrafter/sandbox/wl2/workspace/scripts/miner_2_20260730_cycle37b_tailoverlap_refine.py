"""miner_2 cycle37b: co-crash tail-overlap family, coverage-improved variants.

Cycle37 showed cocrash_spx_60 passes admission gates (IC=-0.0418, ICIR=-0.1129,
maxlibcorr=0.188) but has thin coverage (27% asset-days) because the 5th-pct
threshold with a 60d co-occurrence window requires >=2 crash days in 60d.
This cycle tests coverage-improving parametrizations (same idea, one family):
  - cocrash_spx_60_q10 : 10th-pct threshold (120d est window), 60d co-freq
  - cocrash_spx_90_q10 : 10th-pct threshold (90d est), 90d co-freq
  - cocrash_spx_60_par : parametric threshold mean - 1.0*std (120d), 60d co-freq
  - cocrash_med_60_q10 : cross-median driver at 10th pct (orthogonal proxy)
  - beta_asym_fixed_60 : up-beta minus down-beta, fixed formula (debug from 37)
NOTE: crash_bounce from cycle37 discarded (lookahead bug: shift(-5) inside factor).

Admission gates (10d horizon): abs(IC)>=0.0070, abs(ICIR)>=0.0840.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, per_asset, compute_ic,
                        forward_returns, validate_factor, regime_breakdown,
                        report)

cl = load_close_panel()
print(f"panel {cl.shape}, last date {cl.index[-1].date()}")

ret = cl.pct_change()
med_ret = ret.median(axis=1)
spx_ret = ret["SPX"]


def _crash_ind_q(r, q=0.10, w=120, mp=60):
    th = r.rolling(w, min_periods=mp).quantile(q).shift(1)
    return (r < th).astype(float)


def _crash_ind_par(r, k=1.0, w=120, mp=60):
    m = r.rolling(w, min_periods=mp).mean().shift(1)
    s = r.rolling(w, min_periods=mp).std().shift(1)
    return (r < m - k * s).astype(float)


def excess_cocrash(s, driver, crash_fn, w=60, mp=30, min_marg=0.02):
    ca = crash_fn(s)
    cd = crash_fn(driver)
    joint = (ca * cd).rolling(w, min_periods=mp).mean()
    marg = ca.rolling(w, min_periods=mp).mean()
    base = cd.rolling(w, min_periods=mp).mean()
    out = joint / marg - base
    return out.where(marg >= min_marg)


def beta_asym_fixed(s, driver, w=60, mp=30):
    ar = s.pct_change()
    dr = driver.pct_change()
    up = (dr > 0).astype(float)
    dn = (dr <= 0).astype(float)
    def _b(mask):
        cnt = mask.rolling(w, min_periods=mp).sum()
        exy = (ar * dr * mask).rolling(w, min_periods=mp).sum() / cnt
        ex = (ar * mask).rolling(w, min_periods=mp).sum() / cnt
        ey = (dr * mask).rolling(w, min_periods=mp).sum() / cnt
        eyy = (dr * dr * mask).rolling(w, min_periods=mp).sum() / cnt
        cov = exy - ex * ey
        var = eyy - ey * ey
        b = cov / var
        return b.where((cnt >= mp) & (var.abs() > 1e-12))
    return (_b(up) - _b(dn)).replace([np.inf, -np.inf], np.nan)


cands = {
    "cocrash_spx_60_q10": per_asset(cl, excess_cocrash, spx_ret, _crash_ind_q, 60),
    "cocrash_spx_90_q10": per_asset(cl, excess_cocrash, spx_ret,
                                    lambda r: _crash_ind_q(r, q=0.10, w=90, mp=45), 90),
    "cocrash_spx_60_par": per_asset(cl, excess_cocrash, spx_ret, _crash_ind_par, 60),
    "cocrash_med_60_q10": per_asset(cl, excess_cocrash, med_ret, _crash_ind_q, 60),
    "beta_asym_fixed_60": per_asset(cl, beta_asym_fixed, spx_ret),
}

idx = cl.index
lib = {}
for f in sorted(Path("factors").glob("*.signal.npy")):
    arr = np.load(f)
    if arr.shape == cl.shape:
        fid = f.name.replace(".signal.npy", "")
        lib[fid] = pd.DataFrame(arr, index=idx, columns=cl.columns)
print(f"[lib] loaded {len(lib)} artifacts")

fwd = {str(h): forward_returns(cl, h) for h in (1, 2, 3, 5, 10, 20)}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, cl, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise>0.1:", {k: v for k, v in m.get("library_pairwise_corr", {}).items()
                                if abs(v) > 0.1})
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:22s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/miner_2_20260730_cycle37b_results.json", "w"), indent=1, default=str)
print("\nwrote scripts/miner_2_20260730_cycle37b_results.json")
