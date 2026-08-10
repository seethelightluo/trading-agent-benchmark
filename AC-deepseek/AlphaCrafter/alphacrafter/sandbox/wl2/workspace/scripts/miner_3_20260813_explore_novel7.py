"""miner_3 2026-08-13 exploration batch 7: novel factor candidates (fixed OHLC load).

Candidates:
  1. overnight_drift_20 : mean open/prev-close gap over 20d (gap momentum, complements intraday_drift)
  2. close_pos_20       : mean((close-low)/(high-low), 20) -- where close sits in daily range
  3. vol_term_ratio_5x60: vol5/vol60 raw ratio -- short vs long vol regime slope
  4. dd_recovery_20     : close/rolling_min(close,20)-1 -- bounce off recent 20d low
  5. winrate_20         : fraction of last 20 grid dates asset ret > cross-sectional median ret
  6. updown_vol_ratio_60: std(pos rets)/std(neg rets) over 60d -- asymmetric volatility
  7. variance_ratio_5x20: var(20d)/var(5d*4) -- variance ratio / trending efficiency
  8. max_drawdown_60    : max peak-to-trough drawdown over 60d (deep vs shallow risk)
Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
"""
import json, sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, N_GRID, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)

# ---- raw OHLC loading (grid-aligned) ----
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

fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)

def report(name, mat, extra=""):
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES"); return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    print("=" * 80)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) PASS={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    if ok:
        print("   >>> CANDIDATE PASSES GATE", extra)
    return summ

# 1. overnight drift (open vs prev close)
od = {}
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    od[s] = gap.rolling(20, min_periods=10).mean()
report("overnight_drift_20", to_grid(od))

# 2. close position in daily range (20d mean)
cp = {}
for s, df in series.items():
    hi, lo, cl = df["high"], df["low"], df["close"]
    rng = (hi - lo).replace(0, np.nan)
    pos = safe_div(cl - lo, rng)
    cp[s] = pd.Series(pos, index=df.index).rolling(20, min_periods=10).mean()
report("close_pos_20", to_grid(cp))

# 3. vol term ratio 5/60
vt = {}
for s, df in series.items():
    r = df["ret"]
    v5 = r.rolling(5, min_periods=4).std()
    v60 = r.rolling(60, min_periods=40).std()
    vt[s] = pd.Series(safe_div(v5, v60), index=df.index)
report("vol_term_ratio_5x60", to_grid(vt))

# 4. dd recovery from 20d low
dr = {}
for s, df in series.items():
    c = df["close"]
    mn = c.rolling(20, min_periods=10).min()
    dr[s] = c / mn - 1.0
report("dd_recovery_20", to_grid(dr))

# 5. winrate vs cross-sectional median (grid-level)
ret_mat = to_grid({s: df["ret"] for s, df in series.items()})
med = np.nanmedian(ret_mat, axis=1, keepdims=True)
above = (ret_mat > med).astype(float)
above[~np.isfinite(ret_mat)] = np.nan
wr = np.full_like(above, np.nan)
for j in range(above.shape[1]):
    wr[:, j] = pd.Series(above[:, j]).rolling(20, min_periods=10).mean().values
report("winrate_20", wr)

# 6. up/down vol ratio 60
uv = {}
for s, df in series.items():
    r = df["ret"]
    up = r.where(r > 0, np.nan)
    dn = r.where(r < 0, np.nan)
    up_std = up.rolling(60, min_periods=40).std()
    dn_std = dn.rolling(60, min_periods=40).std()
    uv[s] = safe_div(up_std, dn_std)
report("updown_vol_ratio_60", to_grid(uv))

# 7. variance ratio 5x20 (var20 / (5*var5))
vr = {}
for s, df in series.items():
    r = df["ret"]
    v5 = r.rolling(5, min_periods=4).var()
    v20 = r.rolling(20, min_periods=15).var()
    vr[s] = safe_div(v20, v5 * 5.0)
report("variance_ratio_5x20", to_grid(vr))

# 8. max drawdown 60 (rolling peak-to-trough)
mdd = {}
for s, df in series.items():
    c = df["close"]
    roll_max = c.rolling(60, min_periods=40).max()
    dd = c / roll_max - 1.0
    mdd[s] = dd.rolling(5, min_periods=3).min()  # smoothed min over recent 5d
report("max_drawdown_60", to_grid(mdd))

print("DONE")
