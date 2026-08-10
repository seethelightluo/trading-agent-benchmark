"""miner_3 2026-08-13 exploration: 6 novel factor candidates.
Candidates (all per-asset own-calendar unless noted):
  1. overnight_drift_20  : mean(open/close_prev - 1, 20)  -- gap momentum (complements intraday_drift)
  2. close_pos_20        : mean((close-low)/(high-low), 20) -- where close sits in daily range
  3. vol_term_ratio_5x60 : vol5/vol60 (raw ratio) -- short vs long vol regime slope
  4. dd_recovery_20      : close/rolling_min(close,20)-1 -- bounce off recent 20d low
  5. winrate_20          : fraction of last 20 grid dates asset ret > cross-sectional median ret
  6. updown_vol_ratio_60 : std(pos rets)/std(neg rets) over 60d -- asymmetric volatility
Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
"""
import json, sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (ASSETS, GRID, GIDX, HORIZON, N_GRID, asset_series,
                                  to_grid, cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, roll_mean, roll_std, safe_div)

series = asset_series()
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)

def report(name, mat, extra=""):
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES"); return
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    print("=" * 78)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) PASS={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    if ok:
        print("   >>> CANDIDATE PASSES GATE", extra)
    return summ

# 1. overnight drift
od = {}
for s, df in series.items():
    o = df["open"]; c = df["close"]
    gap = o / c.shift(1) - 1.0
    od[s] = gap.rolling(20, min_periods=10).mean()
mat_od = to_grid(od)
report("overnight_drift_20", mat_od)

# 2. close position in daily range
cp = {}
for s, df in series.items():
    h = df["close"].rolling(20, min_periods=10).mean()  # placeholder
    hi, lo, cl = df["high"], df["low"], df["close"]
    rng = hi - lo
    pos = safe_div(cl - lo, rng)
    pos = pd.Series(pos, index=df.index)
    cp[s] = pos.rolling(20, min_periods=10).mean()
mat_cp = to_grid(cp)
report("close_pos_20", mat_cp)

# 3. vol term ratio 5/60
vt = {}
for s, df in series.items():
    r = df["ret"]
    v5 = r.rolling(5, min_periods=4).std()
    v60 = r.rolling(60, min_periods=40).std()
    vt[s] = safe_div(v5, v60)
mat_vt = to_grid(vt)
report("vol_term_ratio_5x60", mat_vt)

# 4. dd recovery from 20d low
dr = {}
for s, df in series.items():
    c = df["close"]
    mn = c.rolling(20, min_periods=10).min()
    dr[s] = c / mn - 1.0
mat_dr = to_grid(dr)
report("dd_recovery_20", mat_dr)

# 5. winrate vs cross-sectional median (grid-level)
ret_mat = to_grid({s: df["ret"] for s, df in series.items()})
med = np.nanmedian(ret_mat, axis=1, keepdims=True)
above = (ret_mat > med).astype(float)
above[~np.isfinite(ret_mat)] = np.nan
wr = np.full_like(above, np.nan)
for j in range(above.shape[1]):
    s = pd.Series(above[:, j])
    wr[:, j] = s.rolling(20, min_periods=10).mean().values
report("winrate_20", wr)

# 6. up/down vol ratio 60
uv = {}
for s, df in series.items():
    r = df["ret"].dropna()
    idx = r.index
    up = r[r > 0]; dn = r[r < 0]
    up_std = up.rolling(60, min_periods=40).std()
    dn_std = dn.rolling(60, min_periods=40).std()
    ratio = pd.Series(safe_div(up_std.values, dn_std.values), index=idx)
    uv[s] = ratio
mat_uv = to_grid(uv)
report("updown_vol_ratio_60", mat_uv)

print("DONE")
