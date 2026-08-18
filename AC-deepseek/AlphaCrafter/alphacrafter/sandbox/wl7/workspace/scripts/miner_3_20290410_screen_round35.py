"""miner_3 screening round 35: new candidates through 2029-04-09.

Regime: VIX 71 EXTREME-but-easing (-14%/60d), equity bounce (N225 +13.5%/20d,
000300 +7.1%, SPX +4.6%), crypto crushed (ETH -20.3%/20d), WTI weak, US10Y
yields rising (+19.4%/60d). Frozen feeds: 000688.SH, SOX, NDX, CN10Y
(0 returns last 120d) -> live cross-section = 11 assets.

Candidates (h10 admission gate |IC|>=0.0070, |ICIR|>=0.0840):
  A) skew_20d_skip5      : daily-return skewness 20d, skip 5 (asymmetry/tail)
  B) drawdown_depth_60d  : 1 - close/max(close,60) (crash depth, contrarian)
  C) vol_accel_5x20      : 5d realized vol / 20d realized vol (vol spike)
  D) std_rel_mom_20d     : (ret20 - cs median) / cs std of ret20 (standardized rel mom)
  E) vix_beta_cond_60x20 : beta to dVIX * (-VIX 20d mom) (risk-off exposure conditional)
  F) bond_beta_cond_60x20: beta to dUS10Y * US10Y 20d mom (rate-beta conditional)
  G) trend_strength_20   : ret20 / vol20 (vol-normalized trend)
  H) reversal_5d         : -ret5 (short-term reversal)
  I) vol_regime_20x60    : 20d vol / 60d vol (regime change) [sanity vs C]
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, max_lib_corr)

END = "2029-04-09"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
mkt = ret.mean(axis=1)


def skew_20(close, window=20, skip=5, min_periods=12):
    r = close.pct_change().shift(skip)
    return r.rolling(window, min_periods=min_periods).skew()


def drawdown_depth_60(close, window=60, min_periods=30):
    hi = close.rolling(window, min_periods=min_periods).max()
    return 1.0 - close / hi


def vol_accel_5x20(close, swin=5, lwin=20, min_periods=4):
    r = close.pct_change()
    sv = r.rolling(swin, min_periods=min_periods).std()
    lv = r.rolling(lwin, min_periods=min_periods).std()
    return sv / lv


def std_rel_mom_20(close, window=20, skip=5):
    mom = close / close.shift(window + skip) - 1.0
    med = mom.median(axis=1)
    sd = mom.std(axis=1)
    return (mom - med) / sd.replace(0, np.nan)


def vix_beta_cond(close, vix, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    vx = vix.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(vx)
    var = vx.rolling(beta_win, min_periods=min_periods).var()
    beta = cov / var
    vix_mom = vix / vix.shift(cond_win) - 1.0
    return -beta.multiply(vix_mom, axis=0)   # low VIX-beta * VIX easing -> high


def bond_beta_cond(close, us10y, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    y = us10y.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(y)
    var = y.rolling(beta_win, min_periods=min_periods).var()
    beta = cov / var
    y_mom = us10y / us10y.shift(cond_win) - 1.0
    return beta.multiply(y_mom, axis=0)      # high rate-beta * yields rising -> high


def trend_strength_20(close, window=20, min_periods=10):
    r = close.pct_change()
    vol = r.rolling(window, min_periods=min_periods).std()
    mom = close / close.shift(window) - 1.0
    return mom / vol.replace(0, np.nan)


def reversal_5(close, window=5):
    return -(close / close.shift(window) - 1.0)


def vol_regime_20x60(close, swin=20, lwin=60, min_periods=10):
    r = close.pct_change()
    sv = r.rolling(swin, min_periods=min_periods).std()
    lv = r.rolling(lwin, min_periods=min_periods).std()
    return sv / lv


CAND = {
    "skew_20d_skip5": skew_20(close),
    "drawdown_depth_60d": drawdown_depth_60(close),
    "vol_accel_5x20": vol_accel_5x20(close),
    "std_rel_mom_20d": std_rel_mom_20(close),
    "vix_beta_cond_60x20": vix_beta_cond(close, macro["VIX"]),
    "bond_beta_cond_60x20": bond_beta_cond(close, close["US10Y"]),
    "trend_strength_20": trend_strength_20(close),
    "reversal_5d": reversal_5(close),
    "vol_regime_20x60": vol_regime_20x60(close),
}

lib_panels = library_panel(close, macro)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")
print(f"{'candidate':26s} {'IC10':>8s} {'ICIR10':>8s} {'hit10':>6s} {'n':>5s} {'covAD':>7s} {'covD8':>6s} {'turn':>6s} {'maxRho':>7s} {'IC1':>7s} {'IC5':>7s} {'IC20':>7s}")
for name, f in CAND.items():
    summ = {}
    for h in (1, 5, 10, 20):
        ic = daily_ic(f, forward_ret(close, h))
        summ[h] = ic_stats(ic, h)
    s10 = summ[10]
    rho, pairs = max_lib_corr(f, lib_panels)
    cov = coverage_stats(f, forward_ret(close, 10))
    turn = rank_turnover(f, 10)
    print(f"{name:26s} {s10['ic']:8.4f} {s10['icir']:8.3f} {s10['hit']:6.2f} {s10['n']:5d} "
          f"{cov['coverage_asset_days']:7.2f} {cov['coverage_dates_ge8']:6.2f} {turn:6.2f} {rho:7.3f} "
          f"{summ[1]['ic']:7.4f} {summ[5]['ic']:7.4f} {summ[20]['ic']:7.4f}")
    print("   libcorr:", {k: round(v, 3) for k, v in sorted(pairs.items(), key=lambda x: -abs(x[1]))[:4]})

# ---- recent-window stability check (last ~500d) for promising candidates ----
print("\nRecent-window (last 500 trading days) h10 stats:")
for name, f in CAND.items():
    f_r = f.tail(500)
    ic = daily_ic(f_r, forward_ret(close, 10).reindex(f_r.index))
    st = ic_stats(ic, 10)
    print(f"{name:26s} IC10={st['ic']:8.4f} ICIR10={st['icir']:8.3f} hit={st['hit']:5.2f} n={st['n']:4d}")
