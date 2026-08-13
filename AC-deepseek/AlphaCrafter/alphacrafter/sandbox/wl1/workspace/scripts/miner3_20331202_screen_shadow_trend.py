"""miner3 2033-12-02: screen family C - candle shadows & trend/momentum oscillators.
Shadows: upper wick = intraday selling pressure, lower wick = buying pressure.
Trend: medium-horizon momentum (20/60d skip5) and MACD/MA-slope oscillators -
tests whether momentum works at horizons between rev (1-5d) and mom_120d.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner3_eval_lib import load_panel, make_library_factors_full, eval_factor, print_eval

panel = load_panel()
px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']
lib = make_library_factors_full(panel)

def factor_upper_shadow_3d():
    """3d mean upper wick ratio: (high - max(open,close)) / (high-low)"""
    rng = (hi - lo).replace(0, np.nan)
    return ((hi - np.maximum(op, px)) / rng).rolling(3).mean()

def factor_lower_shadow_3d():
    """3d mean lower wick ratio: (min(open,close) - low) / (high-low)"""
    rng = (hi - lo).replace(0, np.nan)
    return ((np.minimum(op, px) - lo) / rng).rolling(3).mean()

def factor_mom_20d_skip5():
    return px.shift(5) / px.shift(25) - 1.0

def factor_mom_60d_skip5():
    return px.shift(5) / px.shift(65) - 1.0

def factor_macd_hist():
    e12 = px.ewm(span=12, adjust=False).mean()
    e26 = px.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    return macd - sig

def factor_trend_slope_20_60():
    """MA20/MA60 - 1: medium trend slope"""
    return px.rolling(20).mean() / px.rolling(60).mean() - 1.0

def factor_body_ratio_3d():
    """3d mean body ratio: (close-open)/(high-low), positive = bullish candle"""
    rng = (hi - lo).replace(0, np.nan)
    return ((px - op) / rng).rolling(3).mean()

cands = {
    'upper_shadow_3d': factor_upper_shadow_3d,
    'lower_shadow_3d': factor_lower_shadow_3d,
    'mom_20d_skip5': factor_mom_20d_skip5,
    'mom_60d_skip5': factor_mom_60d_skip5,
    'macd_hist': factor_macd_hist,
    'trend_slope_20_60': factor_trend_slope_20_60,
    'body_ratio_3d': factor_body_ratio_3d,
}

for name, fn in cands.items():
    try:
        fac = fn()
        res = eval_factor(fac, px, lib=lib)
        print_eval(name, res)
    except Exception as e:
        print(f"ERROR {name}: {e}")
    print()
