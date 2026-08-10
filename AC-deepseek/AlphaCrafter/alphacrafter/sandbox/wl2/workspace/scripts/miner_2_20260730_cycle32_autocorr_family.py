"""miner_2 cycle32: trend-persistence family (return autocorrelation / variance ratio).

Idea: assets whose daily return path shows positive serial correlation (trending
microstructure) may continue trending over the 10d horizon, while mean-reverting
paths may keep chopping. Variance ratio is the multi-period generalization.
These statistics are distinct from level momentum (mom20_volproxy60), path-quality
(gain_loss_20 / intraday_drift_20), and low-vol (calmness_20) already in the library.

Candidates (one family, validated separately):
  - ac1_20   : lag-1 autocorrelation of daily returns, 20d window
  - ac1_60   : lag-1 autocorrelation of daily returns, 60d window
  - vr_60x5  : variance ratio var(5d)/ (5*var(1d)) over trailing 60d
  - vol_accel_5x20 : std5/std20 - 1 (short-term volatility acceleration)
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, compute_ic, forward_returns,
                        validate_factor, library_correlation, regime_breakdown,
                        report)

panel = load_close_panel()

# ---- effective library signals (current active set incl. recovered factors) ----
EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
       "gain_loss_20", "intraday_drift_20"]
lib = {}
idx = panel.index
for fid in EFF:
    a = np.load(Path("factors") / f"{fid}.signal.npy")
    if a.shape[0] == len(idx):
        lib[fid] = pd.DataFrame(a, index=idx, columns=panel.columns)
    else:
        print(f"[lib] shape mismatch {fid}: {a.shape}")

fwd = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}


def ac1(s, w, mp):
    s = s.dropna()
    def _ac1(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 5:
            return np.nan
        a, b = x[:-1], x[1:]
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    return s.pct_change().rolling(w, min_periods=mp).apply(_ac1, raw=True)


def variance_ratio(s, k=5, w=60, mp=30):
    s = s.dropna()
    r1 = s.pct_change()
    rk = s.pct_change(k)
    v1 = r1.rolling(w, min_periods=mp).var()
    vk = rk.rolling(w, min_periods=mp).var()
    return vk / (k * v1)


def per_asset_reindex(panel, func, *a, **kw):
    out = {}
    for c in panel.columns:
        s = panel[c].dropna()
        out[c] = func(s, *a, **kw).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)

cands = {
    "ac1_20": per_asset_reindex(panel, ac1, 20, 10),
    "ac1_60": per_asset_reindex(panel, ac1, 60, 30),
    "vr_60x5": per_asset_reindex(panel, variance_ratio, 5, 60, 30),
    "vol_accel_5x20": per_asset_reindex(panel, lambda s: (s.pct_change().rolling(5, min_periods=3).std()
                                                          / s.pct_change().rolling(20, min_periods=10).std() - 1.0)),
}

print("=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, panel, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    print("    pairwise:", m.get("library_pairwise_corr"))
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:16s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}" for k, v in reg.items()))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner2_cycle32_autocorr_results.json", "w"), indent=1, default=float)
print("DONE")
