"""Screening batch: quickly compute 10d-horizon IC/ICIR for several NEW candidate factor ideas.

Cross-asset 15-instrument universe. Uses factor_common validation window.
This is an exploration screen; promising candidates get dedicated validation scripts.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, forward_returns, rank_ic_series, VAL_START, VAL_END

prices = load_prices(days=2200)
print('loaded assets:', len(prices))
print('date range:', min(df.index.min() for df in prices.values()), '..', max(df.index.max() for df in prices.values()))

CANDIDATES = {}

# A. Bollinger %B / z-score 20d (mean reversion candidate)
def zscore_20(df, s):
    m = df['close'].rolling(20).mean(); sd = df['close'].rolling(20).std()
    return (df['close'] - m) / sd
CANDIDATES['zscore_20'] = zscore_20

# B. RSI 14 (smoothed momentum oscillator)
def rsi_14(df, s):
    d = df['close'].diff()
    up = d.clip(lower=0.0).rolling(14).mean()
    dn = (-d.clip(upper=0.0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)
CANDIDATES['rsi_14'] = rsi_14

# C. Amihud illiquidity 20d (mean |ret|/volume), negative for liquidity preference test
def amihud_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume'].replace(0, np.nan)
    return (r / v).rolling(20).mean()
CANDIDATES['amihud_illiq_20'] = amihud_20

# D. Volume-price correlation 60d: corr(daily ret, volume change)
def vol_price_corr_60(df, s):
    r = df['close'].pct_change()
    vc = df['volume'].pct_change()
    return r.rolling(60).corr(vc)
CANDIDATES['vol_price_corr_60'] = vol_price_corr_60

# E. Garman-Klass vol ratio: 10d efficient vol / 60d efficient vol (vol regime)
def gk_vol_ratio_10x60(df, s):
    hl = np.log(df['high'] / df['low'])
    co = np.log(df['close'] / df['open'])
    gk2 = 0.5 * hl**2 - (2*np.log(2) - 1) * co**2
    gk = np.sqrt(gk2.clip(lower=0))
    v10 = gk.rolling(10).mean(); v60 = gk.rolling(60).mean()
    return v10 / v60.replace(0, np.nan)
CANDIDATES['gk_vol_ratio_10x60'] = gk_vol_ratio_10x60

# F. Return autocorrelation (lag-1) 60d: serial correlation
def autocorr_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
CANDIDATES['autocorr1_60'] = autocorr_60

# G. Risk-adjusted momentum: mom20 / vol20 (Sharpe-style)
def mom20_vol20(df, s):
    m = df['close'].shift(5) / df['close'].shift(25) - 1.0
    v = df['close'].pct_change().rolling(20).std()
    return m / v.replace(0, np.nan)
CANDIDATES['mom20_risk_adj'] = mom20_vol20

# H. 52-week high proximity
def high_prox_252(df, s):
    return df['close'] / df['close'].rolling(252).max()
CANDIDATES['high_prox_252'] = high_prox_252

# I. Gap momentum: mean overnight gap (open vs prev close) over 20d
def gap_mom_20(df, s):
    g = df['open'] / df['close'].shift(1) - 1.0
    return g.rolling(20).mean()
CANDIDATES['gap_mom_20'] = gap_mom_20

# J. Upper shadow ratio 20d (supply pressure)
def upper_shadow_20(df, s):
    hi = df['high']; lo = df['low']; op = df['open']; cl = df['close']
    rng = (hi - lo).replace(0, np.nan)
    shadow = (hi - np.maximum(op, cl)) / rng
    return shadow.rolling(20).mean()
CANDIDATES['upper_shadow_20'] = upper_shadow_20

# K. Conditional momentum: mom10_skip5 scaled by low-vol regime dummy (vol<20d median)
def mom10_cond_lowvol(df, s):
    m = df['close'].shift(5) / df['close'].shift(15) - 1.0
    v = df['close'].pct_change().rolling(20).std()
    lowvol = (v < v.rolling(60).median()).astype(float)
    return m * lowvol
CANDIDATES['mom10_cond_lowvol'] = mom10_cond_lowvol

# L. Downside deviation 20d / total vol 20d (loss aversion proxy)
def downside_ratio_20(df, s):
    r = df['close'].pct_change()
    dd = r.clip(upper=0.0).rolling(20).std()
    td = r.rolling(20).std()
    return dd / td.replace(0, np.nan)
CANDIDATES['downside_ratio_20'] = downside_ratio_20

fwd10 = forward_returns(prices, 10)
results = []
for fid, fn in CANDIDATES.items():
    try:
        panel = factor_to_panel(fn, prices)
        ic = rank_ic_series(panel, fwd10)
        ic = ic[(ic.index >= VAL_START) & (ic.index <= VAL_END)]
        if len(ic) < 100:
            print(f'{fid}: too few IC dates ({len(ic)})'); continue
        mean = float(ic.mean()); std = float(ic.std(ddof=1))
        icir = mean / std if std > 0 else 0.0
        results.append((fid, mean, icir, len(ic), panel.shape))
        print(f'{fid:24s} IC={mean:+.4f} ICIR={icir:+.4f} ndates={len(ic):5d} panel={panel.shape} PASS={abs(mean)>=0.007 and abs(icir)>=0.084}')
    except Exception as e:
        print(f'{fid}: ERROR {e}')

print()
print('=== Sorted by |ICIR| ===')
for fid, mean, icir, n, shp in sorted(results, key=lambda x: -abs(x[2])):
    print(f'{fid:24s} IC={mean:+.4f} ICIR={icir:+.4f} ndates={n}')
