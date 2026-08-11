"""miner_3 screen: quick horizon-10 IC screen for novel factor families (2026-08-13).
Candidates: park_vol_ratio, vol_price_corr, kaufman_eff, ret_autocorr,
upper_shadow, volume_zscore, hurst_rs.
Only quick IC screen; deep validation happens per-candidate afterwards.
"""
import sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from miner3_lib import load_close_panel, rank_ic, WATCHLIST

C, V, H, L, O = load_close_panel(days=4000)
R = C.pct_change()

def park_vol_ratio(win=20):
    # Parkinson vol from high/low vs realized vol from close-close
    park = np.sqrt((np.log(H / L) ** 2).rolling(win).mean() / (4 * math.log(2)))
    rv = R.rolling(win).std()
    return (park / rv).replace([np.inf, -np.inf], np.nan)

def vol_price_corr(win=60):
    # rolling corr between log volume and log price (accumulation/distribution)
    lv = np.log(V.replace(0, np.nan))
    lc = np.log(C)
    out = lv.rolling(win).corr(lc)
    return out

def kaufman_eff(win=20):
    # efficiency ratio: net change / sum of abs moves
    num = (C - C.shift(win)).abs()
    den = C.diff().abs().rolling(win).sum()
    return (num / den).replace([np.inf, -np.inf], np.nan)

def ret_autocorr(lag=1, win=60):
    # rolling lag-1 autocorr of daily returns
    def acf(x):
        x = x.dropna()
        if len(x) < win // 2:
            return np.nan
        x0 = x.iloc[:-lag] if lag > 0 else x
        x1 = x.iloc[lag:]
        if x0.std() == 0 or x1.std() == 0 or len(x0) < win // 2:
            return np.nan
        return np.corrcoef(x0, x1)[0, 1]
    return R.rolling(win).apply(lambda x: acf(pd.Series(x)), raw=False)

def upper_shadow(win=20):
    # average upper shadow fraction of daily range
    rng = (H - L).replace(0, np.nan)
    us = (H - np.maximum(O, C)) / rng
    return us.rolling(win).mean()

def volume_zscore(win=60):
    # recent 20d avg volume vs 60d mean/std
    lv = np.log(V.replace(0, np.nan))
    short = lv.rolling(20).mean()
    mu = lv.rolling(win).mean()
    sd = lv.rolling(win).std()
    return ((short - mu) / sd).replace([np.inf, -np.inf], np.nan)

def hurst_rs(win=120):
    # Hurst exponent via rescaled range on log returns
    lr = np.log(C).diff()
    def h(x):
        x = x.dropna()
        if len(x) < 30:
            return np.nan
        # sub-window R/S
        n = len(x)
        rs = []
        for m in [10, 20, 40, 80]:
            if m < n:
                s = x.iloc[n - m:]
                mean_r = s.mean()
                dev = (s - mean_r).cumsum()
                Rr = dev.max() - dev.min()
                S = s.std(ddof=0)
                if S > 0:
                    rs.append((math.log(m), math.log(Rr / S)))
        if len(rs) >= 3:
            xv = [r[0] for r in rs]; yv = [r[1] for r in rs]
            return np.polyfit(xv, yv, 1)[0]
        return np.nan
    return lr.rolling(win).apply(lambda x: h(pd.Series(x)), raw=False)

cands = {
    'park_vol_ratio_20': park_vol_ratio(20),
    'vol_price_corr_60': vol_price_corr(60),
    'kaufman_eff_20': kaufman_eff(20),
    'ret_autocorr_60x1': ret_autocorr(1, 60),
    'upper_shadow_20': upper_shadow(20),
    'volume_zscore_60': volume_zscore(60),
    'hurst_rs_120': hurst_rs(120),
}

print("=== QUICK SCREEN horizon=10 (research-warm-up data) ===")
for name, panel in cands.items():
    s = rank_ic(panel, R.shift(-10))
    if s is None or len(s) < 30:
        print(f"{name:22s} insufficient IC dates")
        continue
    ic = s.mean(); icir = ic / s.std() if s.std() > 0 else 0.0
    hit = (s > 0).mean()
    cov = panel.notna().sum().sum() / panel.size
    ge8 = (panel.notna().sum(axis=1) >= 8).mean()
    print(f"{name:22s} ic={ic:+.4f} icir={icir:+.3f} hit={hit:.2f} n={len(s):4d} cov={cov:.2f} ge8={ge8:.2f}")
