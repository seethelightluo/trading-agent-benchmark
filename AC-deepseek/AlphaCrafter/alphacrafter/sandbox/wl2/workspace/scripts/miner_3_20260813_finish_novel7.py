"""miner_3 2026-08-13 finish batch 7 (Series fix): complete the 3 candidates
that crashed in explore_novel7 because safe_div returned numpy arrays.

Candidates:
  6. updown_vol_ratio_60: std(pos rets)/std(neg rets) over 60d -- asymmetric vol
  7. variance_ratio_5x20: var(20d)/(5*var(5d)) -- variance ratio / trending efficiency
  8. max_drawdown_60    : rolling peak-to-trough drawdown over 60d (smoothed 5d)
Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)


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


# 6. up/down vol ratio 60  (Series fix applied)
uv = {}
for s, df in series.items():
    r = df["ret"]
    up = r.where(r > 0, np.nan)
    dn = r.where(r < 0, np.nan)
    up_std = up.rolling(60, min_periods=40).std()
    dn_std = dn.rolling(60, min_periods=40).std()
    uv[s] = pd.Series(safe_div(up_std, dn_std), index=df.index)
report("updown_vol_ratio_60", to_grid(uv))

# 7. variance ratio 5x20 (Series fix applied)
vr = {}
for s, df in series.items():
    r = df["ret"]
    v5 = r.rolling(5, min_periods=4).var()
    v20 = r.rolling(20, min_periods=15).var()
    vr[s] = pd.Series(safe_div(v20, v5 * 5.0), index=df.index)
report("variance_ratio_5x20", to_grid(vr))

# 8. max drawdown 60 (rolling peak-to-trough, smoothed 5d min)
mdd = {}
for s, df in series.items():
    c = df["close"]
    roll_max = c.rolling(60, min_periods=40).max()
    dd = c / roll_max - 1.0
    mdd[s] = dd.rolling(5, min_periods=3).min()
report("max_drawdown_60", to_grid(mdd))

print("DONE")
