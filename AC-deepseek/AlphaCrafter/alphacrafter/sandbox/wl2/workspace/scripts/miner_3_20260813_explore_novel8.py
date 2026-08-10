"""miner_3 2026-08-13 exploration batch 8: fresh candidate factors.

Candidates (per-asset own-calendar unless noted):
  1. pct_from_high_252  : close/rolling_max(close,252)-1  -- distance from 252d high
  2. parkinson_vol_20   : sqrt(mean(ln(high/low)^2, 20))  -- range-based vol estimator
  3. reversal_5d        : -(close/close.shift(5)-1)       -- short-term reversal
  4. reversal_20d       : -(close/close.shift(20)-1)      -- medium reversal
  5. ac1_20             : autocorrelation of daily returns over 20d (trend persistence)
  6. xau_beta_cond_60x20: rolling beta vs XAU * (XAU 20d mom) -- gold-beta conditional
  7. wti_beta_cond_60x20: rolling beta vs WTI * (WTI 20d mom) -- energy-beta conditional
  8. amihud_illiq_20    : mean(|ret|/volume, 20)          -- illiquidity proxy (volume assets)
  9. volume_trend_20x60 : mean(vol,20)/mean(vol,60)       -- volume expansion/contraction
 10. rel_mom_20         : 20d ret minus cross-sectional median 20d ret (relative momentum)
Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)


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
    return summ


# 1. distance from 252d high
pfh = {}
for s, df in series.items():
    c = df["close"]
    m = c.rolling(252, min_periods=200).max()
    pfh[s] = pd.Series(c / m - 1.0, index=df.index)
report("pct_from_high_252", to_grid(pfh))

# 2. parkinson vol 20
pk = {}
for s, df in series.items():
    hl = np.log(df["high"] / df["low"])
    pk[s] = pd.Series(np.sqrt((hl ** 2).rolling(20, min_periods=10).mean()), index=df.index)
report("parkinson_vol_20", to_grid(pk))

# 3. reversal 5d
rv5 = {}
for s, df in series.items():
    rv5[s] = pd.Series(-(df["close"] / df["close"].shift(5) - 1.0), index=df.index)
report("reversal_5d", to_grid(rv5))

# 4. reversal 20d
rv20 = {}
for s, df in series.items():
    rv20[s] = pd.Series(-(df["close"] / df["close"].shift(20) - 1.0), index=df.index)
report("reversal_20d", to_grid(rv20))

# 5. autocorrelation of daily returns (20d)
ac1 = {}
for s, df in series.items():
    r = df["ret"]
    ac1[s] = pd.Series(r.rolling(20, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(1), raw=False), index=df.index)
report("ac1_20", to_grid(ac1))
def cond_beta(sym_ret_series, df, window=60, minp=40):
    """rolling beta of df['ret'] vs a benchmark return series, aligned to df index"""
    r = df["ret"]
    bm = sym_ret_series.reindex(r.index)
    cov = r.rolling(window, min_periods=minp).cov(bm)
    var = bm.rolling(window, min_periods=minp).var()
    return pd.Series(safe_div(cov.values, var.values), index=r.index)


# 6. gold-beta conditional
if "XAU" in series:
    xau_ret = series["XAU"]["ret"]
    xau_mom = series["XAU"]["close"] / series["XAU"]["close"].shift(20) - 1.0
    xg = {}
    for s, df in series.items():
        beta = cond_beta(xau_ret, df)
        xg[s] = beta * xau_mom.reindex(df.index)
    report("xau_beta_cond_60x20", to_grid(xg))

# 7. WTI-beta conditional
if "WTI" in series:
    wti_ret = series["WTI"]["ret"]
    wti_mom = series["WTI"]["close"] / series["WTI"]["close"].shift(20) - 1.0
    wg = {}
    for s, df in series.items():
        beta = cond_beta(wti_ret, df)
        wg[s] = beta * wti_mom.reindex(df.index)
    report("wti_beta_cond_60x20", to_grid(wg))

# 8. amihud illiquidity 20
am = {}
for s, df in series.items():
    v = df["volume"]
    if v.abs().sum() == 0:
        continue  # no volume -> NaN coverage
    illiq = pd.Series(safe_div(df["ret"].abs(), v.replace(0, np.nan)), index=df.index)
    am[s] = illiq.rolling(20, min_periods=10).mean()
report("amihud_illiq_20", to_grid(am))

# 9. volume trend 20/60
vt = {}
for s, df in series.items():
    v = df["volume"]
    if v.abs().sum() == 0:
        continue
    v20 = v.rolling(20, min_periods=10).mean()
    v60 = v.rolling(60, min_periods=40).mean()
    vt[s] = pd.Series(safe_div(v20, v60), index=df.index)
report("volume_trend_20x60", to_grid(vt))

# 10. relative momentum 20d (vs cross-sectional median)
rm = {}
for s, df in series.items():
    rm[s] = pd.Series(df["close"] / df["close"].shift(20) - 1.0, index=df.index)
rm_mat = to_grid(rm)
med = np.nanmedian(rm_mat, axis=1, keepdims=True)
rel = rm_mat - med
report("rel_mom_20", rel)

print("DONE")
