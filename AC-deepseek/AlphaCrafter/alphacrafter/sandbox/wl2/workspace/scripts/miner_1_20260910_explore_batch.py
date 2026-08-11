"""miner_1 2026-09-10 exploration batch: novel candidate factors not in library.

Candidates:
 1. gap_mom_10        : cumulative overnight gap returns over 10d (open_t/close_{t-1}-1)
 2. ret_autocorr_60   : lag-1 autocorrelation of signed daily returns over 60d
 3. vol_term_60x20    : vol term structure 60d/20d - 1
 4. volume_trend_20x60: 20d mean volume / 60d mean volume - 1
 5. amihud_20         : mean(|ret|/volume) over 20d (inverted: high = illiquid)
 6. gap_sharpe_20     : mean(gap)/std(gap) over 20d
 7. extreme_freq_20   : fraction of days |ret| > 2*rolling_std(ret,60) over 20d
 8. weekday_mom_60    : mean Monday return over 60d (day-of-week seasonality)

Uses miner_3 shared lib for identical grid/IC convention. Screening only - no persistence.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div, MIN_ASSETS, asset_series)

GATE_IC = 0.0070
GATE_ICIR = 0.0840

series = asset_series()
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
N = len(GRID)
print(f"grid rows: {N} dates {GRID[0]}..{GRID[-1]}")


def roll_mean_s(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()


def roll_std_s(s, w, minp):
    return s.rolling(w, min_periods=minp).std()


def screen(name, cand, verbose=True):
    mat = to_grid(cand)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    q = abs(ic) * abs(icir)
    if verbose:
        print("=" * 90)
        print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
              f"q={q:.5f} cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} GATE={ok}")
        print("   regime:", {k: v for k, v in summ["regime"].items()})
        print("   decay:", dec)
        top = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        print("   top lib corr:", top, "| max_abs:", round(mx_abs, 3), mx_name)
    return {"name": name, "ic": ic, "icir": icir, "q": q, "ok": ok,
            "hit": summ["hit"], "n": summ["n_ic_dates"], "cov_ad": cov_ad,
            "cov_d8": cov_d8, "turn": to, "decay": dec,
            "max_abs_library_correlation": mx_abs, "max_lib_corr_name": mx_name,
            "regime": summ["regime"]}


results = {}

# 1. gap_mom_10: cumulative overnight gap returns over 10d
cand = {}
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    cand[s] = gap.rolling(10, min_periods=6).sum()
results["gap_mom_10"] = screen("gap_mom_10", cand)

# 2. ret_autocorr_60: lag-1 autocorr of signed returns
cand = {}
for s, df in series.items():
    r = df["ret"]
    cand[s] = r.rolling(60, min_periods=40).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
results["ret_autocorr_60"] = screen("ret_autocorr_60", cand)

# 3. vol_term_60x20: 60d vol / 20d vol - 1
cand = {}
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    cand[s] = v60 / v20 - 1.0
results["vol_term_60x20"] = screen("vol_term_60x20", cand)

# 4. volume_trend_20x60: 20d mean volume / 60d mean volume - 1
cand = {}
for s, df in series.items():
    v = df["volume"].astype(float)
    v20 = v.rolling(20, min_periods=10).mean()
    v60 = v.rolling(60, min_periods=30).mean()
    cand[s] = v20 / v60 - 1.0
results["volume_trend_20x60"] = screen("volume_trend_20x60", cand)

# 5. amihud_20: mean(|ret|/volume) over 20d, then negate? raw (low vol per unit vol = liquid)
cand = {}
for s, df in series.items():
    r = df["ret"]
    v = df["volume"].astype(float)
    illiq = (r.abs() / v).rolling(20, min_periods=10).mean()
    cand[s] = illiq
results["amihud_20"] = screen("amihud_20", cand)

# 6. gap_sharpe_20: mean(gap)/std(gap) over 20d
cand = {}
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    cand[s] = safe_div(gap.rolling(20, min_periods=10).mean(), gap.rolling(20, min_periods=10).std())
results["gap_sharpe_20"] = screen("gap_sharpe_20", cand)

# 7. extreme_freq_20: fraction of days |ret| > 2*std(ret,60) within last 20d
cand = {}
for s, df in series.items():
    r = df["ret"]
    sd60 = r.rolling(60, min_periods=30).std()
    extreme = (r.abs() > 2.0 * sd60).astype(float)
    cand[s] = extreme.rolling(20, min_periods=10).mean()
results["extreme_freq_20"] = screen("extreme_freq_20", cand)

# 8. weekday_mom_60: mean Monday return over 60d
cand = {}
for s, df in series.items():
    idx = pd.to_datetime(df.index)
    mon = (idx.dayofweek == 0)
    tmp = df["ret"].copy()
    tmp[~mon] = np.nan
    cand[s] = tmp.rolling(60, min_periods=6).mean()
results["weekday_mom_60"] = screen("weekday_mom_60", cand)

json.dump(results, open("scripts/miner_1_20260910_explore_results.json", "w"), indent=1, default=str)
print("\n=== SUMMARY ===")
for k, v in results.items():
    if v:
        print(f"{k}: IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} q={v['q']:.5f} GATE={v['ok']} "
              f"maxcorr={v['max_abs_library_correlation']:.3f} ({v['max_lib_corr_name']})")
print("DONE")
