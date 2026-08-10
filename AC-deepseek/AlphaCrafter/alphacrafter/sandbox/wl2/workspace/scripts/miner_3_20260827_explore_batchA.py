"""miner_3 2026-08-27 exploration batch A: rotation / relative-strength / trend-dynamics.

Candidates (per-asset own-calendar unless noted):
 1. gr_mom_20         : 20d return minus asset-class mean 20d return (class-neutral rotation)
 2. accel_20_60       : mom20 - mom60 (trend acceleration)
 3. days_since_low_60 : days since close last touched trailing 60d low (recovery stage)
 4. overnight_mom_20  : open-to-open 20d momentum (gap-driven trend; complements intraday_drift_20)
 5. vol_mom5          : vol-scaled 5d reversal: -(mom5 / vol5) (reversal damped by vol)
 6. boll_pos_20x2     : (close - SMA20)/(2*std20) - short-horizon z-score (trend/mean-reversion)
Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)

GROUPS = {
    "equity": ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"],
    "rates": ["US10Y", "CN10Y"],
    "commodity": ["XAU", "COPPER", "WTI"],
    "crypto": ["BTC", "ETH"],
}
GROUP_OF = {a: g for g, aa in GROUPS.items() for a in aa}


def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    df["oret"] = df["open"] / df["open"].shift(1) - 1.0
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}


def report(name, mat, extra=""):
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
    ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    print("=" * 80)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) PASS={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    if ok:
        print("   >>> CANDIDATE PASSES GATE", extra)
    results[name] = {
        "ic": round(ic, 5), "icir": round(icir, 5), "hit": round(summ["hit"], 4),
        "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"],
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10d_rank": round(to, 4), "decay": dec,
        "max_abs_library_correlation": round(mx_abs, 4),
        "max_lib_corr_name": mx_name, "pass_gate": bool(ok),
    }
    return summ


# 1. group-relative momentum 20d
mom20_mat = np.full((len(GRID), len(ASSETS)), np.nan)
for j, s in enumerate(ASSETS):
    if s in series:
        mom20_mat[:, j] = series[s]["close"].pct_change(20).reindex(GRID).values
gr = np.full_like(mom20_mat, np.nan)
for j, s in enumerate(ASSETS):
    g = GROUP_OF.get(s)
    if g is None:
        continue
    members = [s2 for s2 in ASSETS if GROUP_OF.get(s2) == g]
    idx = [ASSETS.index(m) for m in members]
    gr[:, j] = mom20_mat[:, j] - np.nanmean(mom20_mat[:, idx], axis=1)
report("gr_mom_20", gr)

# 2. acceleration: mom20 - mom60
mom60_mat = np.full((len(GRID), len(ASSETS)), np.nan)
for j, s in enumerate(ASSETS):
    if s in series:
        mom60_mat[:, j] = series[s]["close"].pct_change(60).reindex(GRID).values
report("accel_20_60", mom20_mat - mom60_mat)

# 3. days since 60d low
dsl = {}
for s, df in series.items():
    c = df["close"]
    roll_min = c.rolling(60, min_periods=40).min()
    is_low = (c == roll_min)
    days = np.full(len(c), np.nan)
    last = np.nan
    for i in range(len(c)):
        if is_low.iloc[i]:
            last = i
        days[i] = i - last if np.isfinite(last) else np.nan
    dsl[s] = pd.Series(days, index=df.index)
report("days_since_low_60", to_grid(dsl))

# 4. overnight momentum 20d (open-to-open)
om = {}
for s, df in series.items():
    om[s] = pd.Series(df["open"] / df["open"].shift(20) - 1.0, index=df.index)
report("overnight_mom_20", to_grid(om))

# 5. vol-scaled 5d reversal
vm5 = {}
for s, df in series.items():
    r = df["ret"]
    mom5 = df["close"] / df["close"].shift(5) - 1.0
    vol5 = r.rolling(5, min_periods=4).std()
    vm5[s] = pd.Series(-safe_div(mom5, vol5), index=df.index)
report("vol_mom5", to_grid(vm5))

# 6. Bollinger position 20x2
bp = {}
for s, df in series.items():
    c = df["close"]
    mu = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std()
    bp[s] = pd.Series(safe_div(c - mu, 2.0 * sd), index=df.index)
report("boll_pos_20x2", to_grid(bp))

json.dump(results, open("scripts/miner_3_20260827_batchA_results.json", "w"), indent=1)
print("SAVED batchA results")
