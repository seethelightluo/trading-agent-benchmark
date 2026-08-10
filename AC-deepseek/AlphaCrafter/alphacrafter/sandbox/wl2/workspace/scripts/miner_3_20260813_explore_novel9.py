"""miner_3 2026-08-13 exploration batch 9: distinct candidate factor families.

Candidates (per-asset own-calendar unless noted):
  1. zscore_60           : (close - SMA60)/std60 -- standardized price position (trend/mean-reversion hybrid)
  2. range_eff_20        : |20d ret| / sum(20d daily high-low ranges) -- directional efficiency of range
  3. updown_vol_ratio_60 : std(pos rets)/std(neg rets) over 60d -- vol asymmetry
  4. variance_ratio_5x20 : var(20d)/(5*var(5d)) -- Lo-MacKinlay variance ratio (trend efficiency)
  5. max_drawdown_60     : max peak-to-trough drawdown over 60d (deep vs shallow risk, positive=deeper)
  6. gap_ratio_20        : sum(20d overnight gaps)/sum(20d close-close rets) -- where moves happen
  7. us10y_beta_cond_60x20: rolling beta vs US10Y ret * (US10Y 20d mom) -- duration-sensitivity conditional
  8. cn10y_beta_cond_60x20: rolling beta vs CN10Y ret * (CN10Y 20d mom) -- China rate-sensitivity conditional
  9. ndx_beta_cond_60x20 : rolling beta vs NDX ret * (NDX 20d mom) -- tech-gamma conditional
 10. copper_beta_cond_60x20: rolling beta vs COPPER ret * (COPPER 20d mom) -- cyclical exposure conditional
 11. hsi_beta_cond_60x20 : rolling beta vs HSI ret * (HSI 20d mom) -- China-equity sensitivity conditional
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


# 1. z-score of close vs 60d mean
zs = {}
for s, df in series.items():
    c = df["close"]
    mu = c.rolling(60, min_periods=40).mean()
    sd = c.rolling(60, min_periods=40).std()
    zs[s] = pd.Series(safe_div(c - mu, sd), index=df.index)
report("zscore_60", to_grid(zs))

# 2. range efficiency 20d
reff = {}
for s, df in series.items():
    c = df["close"]
    rng = (df["high"] - df["low"]).abs()
    rsum = rng.rolling(20, min_periods=10).sum()
    mom = (c / c.shift(20) - 1.0).abs()
    reff[s] = pd.Series(safe_div(mom, rsum), index=df.index)
report("range_eff_20", to_grid(reff))

# 3. up/down vol ratio 60
ud = {}
for s, df in series.items():
    r = df["ret"]
    pos = r.where(r > 0, np.nan)
    neg = r.where(r < 0, np.nan)
    sp = pos.rolling(60, min_periods=20).std()
    sn = neg.rolling(60, min_periods=20).std()
    ud[s] = pd.Series(safe_div(sp, sn), index=df.index)
report("updown_vol_ratio_60", to_grid(ud))

# 4. variance ratio 5x20 (Lo-MacKinlay style)
vr = {}
for s, df in series.items():
    r = df["ret"]
    v20 = r.rolling(20, min_periods=10).var()
    v5 = r.rolling(5, min_periods=4).var()
    vr[s] = pd.Series(safe_div(v20, 4.0 * v5), index=df.index)
report("variance_ratio_5x20", to_grid(vr))

# 5. max drawdown depth 60d (positive = deeper drawdown)
mdd = {}
for s, df in series.items():
    c = df["close"]
    roll_max = c.rolling(60, min_periods=40).max()
    mdd[s] = pd.Series(1.0 - c / roll_max, index=df.index)
report("max_drawdown_60", to_grid(mdd))

# 6. gap ratio 20 (overnight share of total move)
gr = {}
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    tot = df["close"] / df["close"].shift(1) - 1.0
    gsum = gap.rolling(20, min_periods=10).sum()
    tsum = tot.rolling(20, min_periods=10).sum()
    gr[s] = pd.Series(safe_div(gsum, tsum), index=df.index)
report("gap_ratio_20", to_grid(gr))


def cond_beta(sym_ret_series, df, window=60, minp=40):
    r = df["ret"]
    bm = sym_ret_series.reindex(r.index)
    cov = r.rolling(window, min_periods=minp).cov(bm)
    var = bm.rolling(window, min_periods=minp).var()
    return pd.Series(safe_div(cov.values, var.values), index=r.index)


def cond_report(name, bench_sym):
    if bench_sym not in series:
        print(name, "BENCH MISSING")
        return None
    bm_ret = series[bench_sym]["ret"]
    bm_mom = series[bench_sym]["close"] / series[bench_sym]["close"].shift(20) - 1.0
    out = {}
    for s, df in series.items():
        beta = cond_beta(bm_ret, df)
        out[s] = beta * bm_mom.reindex(df.index)
    report(name, to_grid(out))


# 7-11. cross-asset conditional betas
cond_report("us10y_beta_cond_60x20", "US10Y")
cond_report("cn10y_beta_cond_60x20", "CN10Y")
cond_report("ndx_beta_cond_60x20", "NDX")
cond_report("copper_beta_cond_60x20", "COPPER")
cond_report("hsi_beta_cond_60x20", "HSI")

print("DONE")
