"""miner_1 2026-09-10 batch B: rigorous validation of batchA gate-passers.

Candidates that passed |IC|>=0.007 / |ICIR|>=0.084 in batchA:
  - zsco_20        : (close/sma20-1)/vol20   (vol-scaled distance from 20d mean)
  - zsco_40        : (close/sma40-1)/vol60   (vol-scaled distance from 40d mean)
  - vol_zscore_20  : cross-sectional z-score of 20d realized vol

batchA's library-correlation helper used a single-row Spearman (unreliable -> self-reported
maxlibcorr ~1.0 was an artifact). This script recomputes pairwise rho as the TIME-AVERAGED
cross-sectional Spearman over all overlapping dates (>=8 valid assets per date), which is the
redundancy measure that matters. Also re-derives IC/ICIR, regime split, decay, turnover, coverage.

Data visible through 2026-09-09 (previous completed trading day).
"""
import sys, json, os, glob
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, N_GRID, HORIZON, to_grid, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, coverage_stats, load_asset, load_macro)

# ---------------- data ----------------
series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is not None and len(df) > 120:
        close = df["close"].astype(float)
        ret = close.pct_change()
        series[s] = pd.DataFrame({"close": close, "ret": ret})
print(f"assets loaded: {len(series)}/15")

# ---------------- candidates ----------------
def build_candidates():
    cands = {"zsco_20": {}, "zsco_40": {}, "vol_zscore_20": {}}
    vol20_mat = np.full((N_GRID, len(ASSETS)), np.nan)
    for i, s in enumerate(ASSETS):
        d = series[s]
        close, ret = d["close"], d["ret"]
        vol20 = ret.rolling(20).std()
        vol60 = ret.rolling(60).std()
        sma20 = close.rolling(20).mean()
        sma40 = close.rolling(40).mean()
        cands["zsco_20"][s] = (close / sma20 - 1.0) / vol20
        cands["zsco_40"][s] = (close / sma40 - 1.0) / vol60
        vol20_mat[:, i] = vol20.reindex(GRID).values
    # cross-sectional z-score of 20d vol (row-wise)
    zmat = np.full_like(vol20_mat, np.nan)
    for t in range(N_GRID):
        row = vol20_mat[t]
        ok = ~np.isnan(row)
        if ok.sum() >= 8:
            m = np.nanmean(row[ok]); sd = np.nanstd(row[ok])
            if sd > 1e-12:
                zmat[t, ok] = (row[ok] - m) / sd
    cands["vol_zscore_20"] = {s: pd.Series(zmat[:, i], index=GRID) for i, s in enumerate(ASSETS)}
    return cands

# ---------------- proper library correlation ----------------
def library_corr_timeavg(factor_mat, min_dates=60):
    """Time-averaged cross-sectional Spearman rho vs every factors/*.signal.npy artifact."""
    ours = cross_sectional_rank(factor_mat)  # (N, 15) ranks 0..1
    out = {}
    for f in sorted(glob.glob("factors/*.signal.npy")):
        arr = np.load(f, allow_pickle=True)
        rows = min(arr.shape[0], ours.shape[0])
        a = ours[:rows]
        b = cross_sectional_rank(arr[:rows])
        rhos = []
        for t in range(rows):
            x = a[t]; y = b[t]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 8:
                xs = pd.Series(x[ok]).rank(); ys = pd.Series(y[ok]).rank()
                c = xs.corr(ys)
                if np.isfinite(c):
                    rhos.append(c)
        if len(rhos) >= min_dates:
            out[os.path.basename(f).replace(".signal.npy", "")] = {
                "mean_rho": round(float(np.mean(rhos)), 4),
                "median_rho": round(float(np.median(rhos)), 4),
                "n_dates": len(rhos),
                "pct_abs_gt_05": round(float(np.mean(np.abs(rhos) > 0.5)), 4),
            }
    return out

# ---------------- evaluate ----------------
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}

def report(name, mat):
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(f"{name}: NO VALID IC DATES"); return None
    cov_ad, cov_d8 = coverage_stats(mat)
    turn = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    libc = library_corr_timeavg(mat)
    top = sorted(libc.items(), key=lambda kv: abs(kv[1]["mean_rho"]), reverse=True)[:5]
    ic, icir = summ["ic"], summ["icir"]
    gate_ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    top_rho = top[0][1]["mean_rho"] if top else 0.0
    print("=" * 110)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={turn:.3f} | GATE_IC={'PASS' if gate_ok else 'FAIL'}")
    print("  regime:", {k: f"{v['ic']:+.3f}/{v['icir']:+.2f}(n={v['n']})" for k, v in summ["regime"].items()})
    print("  decay:", dec)
    print("  top library corr (time-avg Spearman):")
    for fn, v in top:
        print(f"     {fn:28s} mean_rho={v['mean_rho']:+.3f} med={v['median_rho']:+.3f} n={v['n_dates']} pct|r|>0.5={v['pct_abs_gt_05']:.3f}")
    return {"ic": ic, "icir": icir, "hit": summ["hit"], "n": summ["n_ic_dates"],
            "cov_ad": cov_ad, "cov_d8": cov_d8, "turn": turn, "decay": dec,
            "regime": summ["regime"], "gate_ok": gate_ok,
            "top_libcorr": top, "max_abs_mean_rho": top_rho}

cands = build_candidates()
for name, cd in cands.items():
    mat = to_grid(cd)
    results[name] = report(name, mat)

print("\n===== SUMMARY =====")
for k, v in results.items():
    if v is None:
        continue
    print(f"{k:16s} IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} n={v['n']} turn={v['turn']:.3f} "
          f"max_abs_mean_rho={v['max_abs_mean_rho']:.3f} GATE_IC={'PASS' if v['gate_ok'] else '--'}")

json.dump(results, open("scripts/miner_1_20260910_batchB_results.json", "w"), indent=1, default=str)
print("saved scripts/miner_1_20260910_batchB_results.json")
