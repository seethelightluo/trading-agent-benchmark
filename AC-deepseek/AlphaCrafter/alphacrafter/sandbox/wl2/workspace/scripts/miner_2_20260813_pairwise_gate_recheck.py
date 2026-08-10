"""miner_2 2026-08-13: pairwise correlation gate re-check for sharpe_20 / drawup_20.

Both passed the IC/ICIR admission gates (sharpe_20 IC +0.0426 ICIR +0.128;
drawup_20 IC +0.0460 ICIR +0.133) but were evicted in the cycle-37 audit for
pairwise correlation conflicts with active library members
(downbeta_spx_60, mom20_volproxy60, usdjpy_beta_cond_120x60; |rho| 0.57-0.71).
Re-admission requires re-running the pairwise gate against the CURRENT active
library, computed from the real signal artifacts (factors/*.signal.npy).
"""
import numpy as np
import pandas as pd
import json
import glob
import os

MIN_ASSETS = 8

ACTIVE_W_NPY = [
    "calmness_20", "downbeta_spx_60", "dxy_beta_cond_60x20", "lagbeta_spx_60",
    "mom20_volproxy60", "ret_skew_10", "usdjpy_beta_cond_120x60", "volcluster_60",
]
CANDIDATES = ["sharpe_20", "drawup_20"]

# active list from root JSON status scan (EFFECTIVE + recoverable npy artifact)
active = []
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f or "ensemble" in f or "audit" in f:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("validation", {}).get("status") == "EFFECTIVE":
        art = d.get("signal_artifact", "")
        if art and os.path.exists("factors/" + art):
            active.append(os.path.basename(f).replace(".json", ""))
active = sorted(set(active))
print("ACTIVE library factors with recoverable npy artifact (%d):" % len(active))
print(active)


def load(nm):
    return np.load(f"factors/{nm}.signal.npy", allow_pickle=True).astype(float)


def daily_spearman(a, b):
    """per-date cross-sectional Spearman rho; returns list of rhos over dates."""
    rhos = []
    n = min(a.shape[0], b.shape[0])
    a = a[-n:]
    b = b[-n:]
    for t in range(n):
        x, y = a[t], b[t]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < MIN_ASSETS:
            continue
        xs = pd.Series(x[ok]).rank()
        ys = pd.Series(y[ok]).rank()
        c = xs.corr(ys)
        if np.isfinite(c):
            rhos.append(c)
    return np.array(rhos)


print("\n=== PAIRWISE GATE (Spearman, mean over dates) ===")
for cand in CANDIDATES:
    ca = load(cand)
    results = {}
    for lib in active:
        if lib == cand:
            continue
        la = load(lib)
        rhos = daily_spearman(ca, la)
        if len(rhos) == 0:
            continue
        mean_rho = float(np.mean(rhos))
        last60 = float(np.mean(rhos[-60:])) if len(rhos) >= 60 else float("nan")
        results[lib] = {
            "mean_rho": round(mean_rho, 4),
            "median_rho": round(float(np.median(rhos)), 4),
            "freq_gt_05": round(float((np.abs(rhos) > 0.5).mean()), 4),
            "last60": round(last60, 4),
            "n_dates": len(rhos),
        }
    conflicts = {k: v for k, v in results.items() if abs(v["mean_rho"]) > 0.5}
    mx = max(results.items(), key=lambda kv: abs(kv[1]["mean_rho"])) if results else (None, None)
    print(f"\n[{cand}] max|mean_rho| = {abs(mx[1]['mean_rho']) if mx[1] else 'NA'} vs {mx[0] if mx else 'NA'}")
    for k, v in sorted(results.items(), key=lambda kv: -abs(kv[1]["mean_rho"])):
        flag = " <== CONFLICT" if abs(v["mean_rho"]) > 0.5 else ""
        print(f"  {k:26s} mean={v['mean_rho']:+.4f} med={v['median_rho']:+.4f} "
              f"last60={v['last60']:+.4f} n={v['n_dates']}{flag}")
    print(f"  CONFLICTS>0.5: {list(conflicts.keys()) if conflicts else 'NONE'}")
