"""miner_2 cycle37: tail-risk overlap / co-crash coupling family.

Rationale: active library covers level momentum (mom_20d_skip5, mom20_volproxy60),
vol level (calmness_20, volcluster_60), downside beta to SPX (downbeta_spx_60),
freshness (days_since_high_60), streaks, range position. None of the active
factors model *joint tail co-occurrence* (how coupled an asset's crash risk is
to the cross-asset aggregate) or *drawdown-adjusted momentum* (momentum priced
per unit of realized drawdown depth). These are conditional-risk ideas distinct
from raw vol and from plain beta.

Candidates (one family: tail-overlap & drawdown-conditional momentum):
  - cocrash_spx_60 : P(SPX crash | asset crash) - P(SPX crash) over 60d
                     (excess tail coupling to SPX; 120d rolling 5th-pct threshold)
  - cocrash_med_60 : same but market proxy = cross-asset median return
                     (orthogonal-ish to downbeta_spx_60 which uses SPX beta)
  - beta_asym_60   : up-market beta - down-market beta over 60d (beta asymmetry;
                     downbeta_spx_60 is only the downside half)
  - mdd_adj_mom_20x60 : 20d momentum / (1 + 60d max drawdown) (risk-adj momentum;
                     distinct from vol-scaled mom20_volproxy60)
  - crash_bounce_60x5 : mean 5d forward return after joint asset+SPX crash days
                     minus unconditional 5d mean (rebound/resilience premium)

Admission gates (10d horizon): abs(IC) >= 0.0070, abs(ICIR) >= 0.0840,
max_abs_library_correlation < 0.5 (reported as audit provenance).
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
med_ret = ret.median(axis=1)          # cross-asset aggregate proxy
spx_ret = ret["SPX"]

Q = 0.05                               # tail threshold quantile
TH = 120                               # threshold estimation window


def _crash_ind(r: pd.Series, th_q: float = Q, w: int = TH, mp: int = 60):
    """1 where return < rolling w-day quantile (threshold known at t-1)."""
    th = r.rolling(w, min_periods=mp).quantile(th_q).shift(1)
    return (r < th).astype(float)


def excess_cocrash(s, driver, w=60, mp=30):
    """P(driver crash | asset crash) - P(driver crash), rolling w."""
    ca = _crash_ind(s)
    cd = _crash_ind(driver)
    joint = (ca * cd).rolling(w, min_periods=mp).mean()
    marg = ca.rolling(w, min_periods=mp).mean()
    base = cd.rolling(w, min_periods=mp).mean()
    out = joint / marg - base
    return out.where(marg >= 0.02)     # require non-degenerate marginal


def crash_bounce(s, driver, fwd_h=5, w=60, mp=30):
    """Mean fwd_h-day forward return after joint crash days - unconditional mean."""
    ca = _crash_ind(s)
    cd = _crash_ind(driver)
    joint = (ca * cd).astype(float)
    fwd = s.shift(-fwd_h) / s - 1.0
    cond = (fwd * joint).rolling(w, min_periods=mp).sum() / joint.rolling(w, min_periods=mp).sum()
    uncond = fwd.rolling(w, min_periods=mp).mean()
    return (cond - uncond).where(joint.rolling(w, min_periods=mp).sum() >= 2)


def beta_asym(s, driver, w=60, mp=30):
    """Up-market beta minus down-market beta (sign flipped: positive = less crash-sensitive)."""
    ar = s.pct_change()
    dr = driver.pct_change()
    up = (dr > 0).astype(float)
    dn = (dr <= 0).astype(float)
    cov_up = (ar * dr * up).rolling(w, min_periods=mp).sum() / up.rolling(w, min_periods=mp).sum()
    var_up = (dr * dr * up).rolling(w, min_periods=mp).sum() / up.rolling(w, min_periods=mp).sum()
    cov_dn = (ar * dr * dn).rolling(w, min_periods=mp).sum() / dn.rolling(w, min_periods=mp).sum()
    var_dn = (dr * dr * dn).rolling(w, min_periods=mp).sum() / dn.rolling(w, min_periods=mp).sum()
    b_up = cov_up / var_up
    b_dn = cov_dn / var_dn
    return (b_up - b_dn).replace([np.inf, -np.inf], np.nan)


def max_dd_60(s, w=60, mp=30):
    """Max peak-to-trough drawdown over rolling w (positive number)."""
    def _mdd(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        peak = np.maximum.accumulate(x)
        dd = (x - peak) / peak
        return float(-dd.min())
    return s.rolling(w, min_periods=mp).apply(_mdd, raw=True)


def mdd_adj_mom(s, mom_w=20, dd_w=60, mp=30):
    mom = s / s.shift(mom_w) - 1.0
    dd = max_dd_60(s, dd_w, mp)
    return mom / (1.0 + dd)


cands = {
    "cocrash_spx_60": per_asset(cl, excess_cocrash, spx_ret),
    "cocrash_med_60": per_asset(cl, excess_cocrash, med_ret),
    "beta_asym_60": per_asset(cl, beta_asym, spx_ret),
    "mdd_adj_mom_20x60": per_asset(cl, mdd_adj_mom),
    "crash_bounce_60x5": per_asset(cl, crash_bounce, spx_ret),
}

# ---- library: all real .signal.npy artifacts with matching shape ----
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
    print(f"  {name:20s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/miner_2_20260730_cycle37_results.json", "w"), indent=1, default=str)
print("\nwrote scripts/miner_2_20260730_cycle37_results.json")
