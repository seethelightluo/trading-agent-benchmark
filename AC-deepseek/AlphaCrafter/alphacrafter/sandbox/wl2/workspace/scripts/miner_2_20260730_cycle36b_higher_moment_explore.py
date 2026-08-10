"""miner_2 cycle36b: higher-moment (return distribution shape) family.

Rationale: library covers level momentum, vol (calmness), streaks, recovery
freshness, macro-beta. None of the active factors model the *shape* of the
daily-return distribution (skewness / kurtosis). In single-asset literature,
negative-skew (lottery-like) and fat-tailed assets tend to underperform on a
risk-adjusted basis; testing whether this generalizes to the 15-asset
cross-asset universe.

Candidates (one family: distribution shape):
  - skew_60  : rolling Pearson skewness of daily returns over 60d
  - kurt_60  : rolling excess kurtosis of daily returns over 60d
  - coskew_60: rolling skewness of returns on days when cross-asset median
               return is negative (downside skewness) -- crash-sensitive half
  - pos_skew_ratio_60: skew computed on positive returns only (upside skew)

Admission gates (10d horizon): abs(IC) >= 0.0070, abs(ICIR) >= 0.0840,
max_abs_library_correlation < 0.5.
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

# cross-asset median return series (for downside conditioning)
ret = cl.pct_change()
med_ret = ret.median(axis=1)


def skew_series(s, w=60, mp=30):
    r = s.pct_change()
    return r.rolling(w, min_periods=mp).skew()


def kurt_series(s, w=60, mp=30):
    r = s.pct_change()
    return r.rolling(w, min_periods=mp).kurt()


def downside_skew(s, w=60, mp=30, med=med_ret):
    def _sk(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        x = x - x.mean()
        sd = x.std()
        if sd <= 0:
            return np.nan
        return float((x ** 3).mean() / sd ** 3)
    r = s.pct_change()
    # only keep days where cross-asset median return < 0
    mask = (med.reindex(r.index) < 0).astype(float)
    rr = r.where(mask > 0)
    return rr.rolling(w, min_periods=mp).apply(_sk, raw=True)


def upside_skew(s, w=60, mp=30):
    def _sk(x):
        x = np.asarray(x, dtype=float)
        if len(x) < mp:
            return np.nan
        x = x - x.mean()
        sd = x.std()
        if sd <= 0:
            return np.nan
        return float((x ** 3).mean() / sd ** 3)
    r = s.pct_change()
    rr = r.where(r > 0)
    return rr.rolling(w, min_periods=mp).apply(_sk, raw=True)


cands = {
    "skew_60": per_asset(cl, skew_series),
    "kurt_60": per_asset(cl, kurt_series),
    "downside_skew_60": per_asset(cl, downside_skew),
    "upside_skew_60": per_asset(cl, upside_skew),
}

# ---- library: all real .signal.npy artifacts with matching shape ----
idx = cl.index
lib = {}
for f in sorted(Path("factors").glob("*.signal.npy")):
    arr = np.load(f)
    if arr.shape == cl.shape:
        fid = f.name.replace(".signal.npy", "")
        if fid != "downside_dev_60":
            lib[fid] = pd.DataFrame(arr, index=idx, columns=cl.columns)
print(f"[lib] loaded {len(lib)} artifacts")

fwd = {str(h): forward_returns(cl, h) for h in (1, 2, 3, 5, 10, 20)}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, cl, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise:", {k: v for k, v in m.get("library_pairwise_corr", {}).items()
                            if abs(v) > 0.1})
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:18s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/miner_2_20260730_cycle36b_results.json", "w"), indent=1, default=str)
print("\nwrote scripts/miner_2_20260730_cycle36b_results.json")
