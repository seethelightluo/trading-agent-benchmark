"""miner3 2033-12-02: screen family B - return asymmetry & volatility structure.
Motivation: high-dispersion regime; asymmetry/tail and short/long vol-ratio factors
capture crash-risk and vol-mean-reversion dimensions not covered by library
(vol_of_vol20x60 = vol of vol; these are vol-ratio & asymmetry).
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner3_eval_lib import load_panel, make_library_factors_full, eval_factor, print_eval

panel = load_panel()
px = panel['close']; ret = panel['ret']
lib = make_library_factors_full(panel)

def factor_downside_vol_ratio_20():
    """downside semideviation / total std over 20d. High = bad-tail asymmetry."""
    mu = ret.mean(axis=0)
    dn = ret.clip(upper=0)
    dvol = np.sqrt((dn ** 2).rolling(20).mean())
    tvol = ret.rolling(20).std()
    return dvol / tvol

def factor_skew_20d():
    """rolling 20d skewness of daily returns (high = right tail / lottery)"""
    n = 20
    mu = ret.rolling(n).mean()
    sd = ret.rolling(n).std()
    m3 = ((ret - mu) ** 3).rolling(n).mean()
    return m3 / sd ** 3

def factor_vol_ratio_5_60():
    """5d realized vol / 60d realized vol (short-term vol expansion)"""
    return ret.rolling(5).std() / ret.rolling(60).std()

def factor_vol_ratio_10_60():
    """10d realized vol / 60d realized vol"""
    return ret.rolling(10).std() / ret.rolling(60).std()

def factor_minret_20d():
    """-min daily return over 20d (worst-day shock, reversal prone)"""
    return -ret.rolling(20).min()

def factor_maxret_20d():
    """max daily return over 20d (best-day euphoria, reversal prone)"""
    return ret.rolling(20).max()

cands = {
    'downside_vol_ratio_20': factor_downside_vol_ratio_20,
    'skew_20d': factor_skew_20d,
    'vol_ratio_5_60': factor_vol_ratio_5_60,
    'vol_ratio_10_60': factor_vol_ratio_10_60,
    'minret_20d': factor_minret_20d,
    'maxret_20d': factor_maxret_20d,
}

for name, fn in cands.items():
    try:
        fac = fn()
        res = eval_factor(fac, px, lib=lib)
        print_eval(name, res)
    except Exception as e:
        print(f"ERROR {name}: {e}")
    print()
