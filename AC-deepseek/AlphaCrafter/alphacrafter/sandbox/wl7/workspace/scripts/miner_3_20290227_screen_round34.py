"""miner_3 screening round 34: new candidates through 2029-02-26 (deep-bear regime).

Candidates (h10 admission gate |IC|>=0.0070, |ICIR|>=0.0840):
  A) downside_beta_60d     : beta to cross-sectional market return on DOWN days only
  B) rsi_14d               : classic RSI(14) (mean-reversion)
  C) parkinson_vol_20d     : high-low range vol estimator (OHLC, distinct from std vol)
  D) vol_trend_20x60       : 20d realized vol / 60d realized vol (vol regime change)
  E) cvar_20d              : 5% CVaR of daily returns over 20d (tail loss)
  F) usdcny_beta_cond_60x20: beta to USDCNY * USDCNY 20d mom (CNY weakness proxy)
  G) close_pos_20d         : (close - min20)/(max20 - min20) position in range
  H) macd_12_26_norm       : (EMA12 - EMA26)/close

NOTE: 000688.SH, SOX, NDX, CN10Y feeds frozen (0 returns last 130d); live cross-section = 11 assets.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, max_lib_corr)

END = "2029-02-26"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
mkt = ret.mean(axis=1)

# ---- A) downside beta ----
def downside_beta_60(close, window=60, min_periods=30):
    r = close.pct_change()
    mk = r.mean(axis=1)
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for a in r.columns:
        cov = r[a].rolling(window, min_periods=min_periods).cov(mk)
        var = mk.rolling(window, min_periods=min_periods).var()
        out[a] = cov / var
    return out

# ---- B) RSI 14 ----
def rsi_14(close, window=14):
    r = close.pct_change()
    up = r.clip(lower=0).rolling(window).mean()
    dn = (-r.clip(upper=0)).rolling(window).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

# ---- C) parkinson vol ----
def parkinson_vol_20(close, window=20, min_periods=10):
    # use high/low via close-range proxy? We only loaded close. Load OHLC here instead.
    return None

# ---- D) vol trend ----
def vol_trend_20x60(close, swin=20, lwin=60, min_periods=10):
    r = close.pct_change()
    sv = r.rolling(swin, min_periods=min_periods).std()
    lv = r.rolling(lwin, min_periods=min_periods).std()
    return sv / lv

# ---- E) CVaR 5% ----
def cvar_20(close, window=20, min_periods=10, q=0.05):
    r = close.pct_change()
    def cv(x):
        if len(x) < min_periods:
            return np.nan
        return -np.percentile(x, 100 * q)
    return r.rolling(window, min_periods=min_periods).apply(cv, raw=True)

# ---- F) USDCNY conditional beta ----
def usdcny_beta_cond(close, usdcny, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    fx = usdcny.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(fx)
    var = fx.rolling(beta_win, min_periods=min_periods).var()
    beta = cov / var
    fx_mom = usdcny / usdcny.shift(cond_win) - 1.0
    return -beta.multiply(fx_mom, axis=0)   # negative: low CNY-beta * CNY weakness -> high

# ---- G) close position ----
def close_pos_20(close, window=20, min_periods=10):
    hi = close.rolling(window, min_periods=min_periods).max()
    lo = close.rolling(window, min_periods=min_periods).min()
    return (close - lo) / (hi - lo)

# ---- H) MACD norm ----
def macd_norm(close, fast=12, slow=26, min_periods=10):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    return (ef - es) / close

CAND = {
    "downside_beta_60d": downside_beta_60(close),
    "rsi_14d": rsi_14(close),
    "vol_trend_20x60": vol_trend_20x60(close),
    "cvar_20d": cvar_20(close),
    "usdcny_beta_cond_60x20": usdcny_beta_cond(close, macro["USDCNY"]),
    "close_pos_20d": close_pos_20(close),
    "macd_12_26_norm": macd_norm(close),
}

lib_panels = library_panel(close, macro)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")
print(f"{'candidate':26s} {'IC10':>8s} {'ICIR10':>8s} {'hit10':>6s} {'n':>5s} {'covAD':>7s} {'covD8':>6s} {'turn':>6s} {'maxRho':>7s} {'IC1':>7s} {'IC5':>7s} {'IC20':>7s}")
for name, f in CAND.items():
    if f is None:
        continue
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
