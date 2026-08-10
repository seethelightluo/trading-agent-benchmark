"""miner_3 2026-08-13: strict active-library correlation check.

Candidates that passed |IC|/|ICIR| gate in batch7:
  - dd_recovery_20 (IC +0.0399, ICIR +0.1159)
  - winrate_20     (IC +0.0255, ICIR +0.0841)
Check daily cross-sectional Spearman rho vs the 8 LIVE ensemble factor
artifacts (as-consumed transform: cross-sectional rank). Report full-period
mean/median/std rho, |rho|>0.5 freq, last-60d mean, and the 8 candidates'
library admission contract (|rho|<0.5 vs any live factor, full & last60).
"""
import sys
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
    """daily cross-sectional Spearman rho between two rank matrices."""
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


for cname, cmat in cands.items():
    print("=" * 80)
    print(f"CANDIDATE {cname}: daily rank-rho vs LIVE ensemble factors")
    worst = None
    for live in LIVE:
        art = np.load(f"factors/{live}.signal.npy", allow_pickle=True)
        rows = min(art.shape[0], cmat.shape[0])
        live_rank = cross_sectional_rank(art[:rows])
        rho = daily_rho(cmat[:rows], live_rank)
        valid = rho[~np.isnan(rho)]
        if len(valid) < 30:
            print(f"  {live}: insufficient overlap ({len(valid)})")
            continue
        m = float(np.mean(valid))
        md = float(np.median(valid))
        s = float(np.std(valid))
        frac50 = float(np.mean(np.abs(valid) > 0.5))
        last60 = float(np.mean(valid[-60:])) if len(valid) >= 60 else float("nan")
        print(f"  {live:22s} mean={m:+.3f} med={md:+.3f} sd={s:.3f} |rho|>0.5:{frac50:.3f} last60={last60:+.3f} n={len(valid)}")
        if worst is None or abs(m) > abs(worst[1]):
            worst = (live, m, frac50, last60)
    if worst:
        print(f"  >>> WORST live: {worst[0]} mean rho={worst[1]:+.3f} |rho|>0.5 freq={worst[2]:.3f} last60={worst[3]:+.3f}")

print("DONE")
