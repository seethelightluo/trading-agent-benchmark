"""miner_3 2026-08-13: strict active-library correlation check (fixed: skip missing artifacts).

Candidates that passed |IC|/|ICIR| gate in batch7:
  - dd_recovery_20 (IC +0.0399, ICIR +0.1159)
  - winrate_20     (IC +0.0255, ICIR +0.0841)
Check daily cross-sectional Spearman rho vs LIVE ensemble factor artifacts
(as-consumed transform: cross-sectional rank). Report full-period mean/median/std rho,
|rho|>0.5 freq, last-60d mean. Admission contract: |rho|<0.5 vs any live factor.
"""
import sys, os
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, N_GRID, to_grid,
                                  cross_sectional_rank, safe_div, MIN_ASSETS)

LIVE = ["downbeta_spx_60", "mom20_volproxy60", "mom_20d_skip5", "gain_loss_20",
        "usdjpy_beta_cond_120x60", "volcluster_60", "range_pos_252", "calmness_20"]


def load_asset(sym, days=2200):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}

# candidate raw matrices
dr = {}
for s, df in series.items():
    c = df["close"]
    mn = c.rolling(20, min_periods=10).min()
    dr[s] = c / mn - 1.0
dd_rec = cross_sectional_rank(to_grid(dr))

ret_mat = to_grid({s: df["ret"] for s, df in series.items()})
med = np.nanmedian(ret_mat, axis=1, keepdims=True)
above = (ret_mat > med).astype(float)
above[~np.isfinite(ret_mat)] = np.nan
wr = np.full_like(above, np.nan)
for j in range(above.shape[1]):
    wr[:, j] = pd.Series(above[:, j]).rolling(20, min_periods=10).mean().values
winrate = cross_sectional_rank(wr)

cands = {"dd_recovery_20": dd_rec, "winrate_20": winrate}


def daily_rho(a, b):
    T = a.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        x, y = a[t], b[t]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < MIN_ASSETS:
            continue
        xs = pd.Series(x[ok]).rank()
        ys = pd.Series(y[ok]).rank()
        c = xs.corr(ys)
        if np.isfinite(c):
            out[t] = c
    return out


summary = {}
for cname, cmat in cands.items():
    print("=" * 80)
    print(f"CANDIDATE {cname}: daily rank-rho vs LIVE ensemble factors")
    worst = None
    for live in LIVE:
        p = f"factors/{live}.signal.npy"
        if not os.path.exists(p):
            print(f"  {live:24s} MISSING artifact - skipped")
            continue
        art = np.load(p, allow_pickle=True)
        rows = min(art.shape[0], cmat.shape[0])
        live_rank = cross_sectional_rank(art[:rows])
        rho = daily_rho(cmat[:rows], live_rank)
        valid = rho[~np.isnan(rho)]
        if len(valid) < 50:
            print(f"  {live:24s} too few valid dates ({len(valid)})")
            continue
        mean_r, med_r, sd_r = float(valid.mean()), float(np.median(valid)), float(valid.std())
        f_gt = float(np.mean(np.abs(valid) > 0.5))
        last60 = float(valid[-60:].mean()) if len(valid) >= 60 else float("nan")
        n = len(valid)
        print(f"  {live:24s} mean={mean_r:+.3f} med={med_r:+.3f} sd={sd_r:.3f} "
              f"|rho|>0.5:{f_gt:.3f} last60={last60:+.3f} n={n}")
        summary.setdefault(cname, {})[live] = {"mean": mean_r, "med": med_r, "sd": sd_r,
                                               "freq_gt_05": f_gt, "last60": last60, "n": n}
        if worst is None or abs(mean_r) > abs(worst[1]):
            worst = (live, mean_r)
    if worst:
        verdict = "PASS(corr)" if abs(worst[1]) < 0.5 else "FAIL(corr)"
        print(f"  -> worst live rho: {worst[0]} {worst[1]:+.3f}  VERDICT: {verdict}")
    else:
        print("  -> no live artifacts compared")

with open("scripts/miner_3_20260813_active_corr_results.json", "w") as f:
    json.dump(summary, f, indent=1)
print("saved scripts/miner_3_20260813_active_corr_results.json")
