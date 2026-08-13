"""miner3 2033-11-04: screen volume-price interaction factors (family A).
Volume data exists for equity indices + crypto (9/15 assets). Evaluate IC/ICIR/turnover/coverage/rho vs library.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner3_eval_lib import load_panel, make_library_factors_full, eval_factor, print_eval

panel = load_panel()
px = panel['close']; vol = panel['vol']; ret = panel['ret']
lib = make_library_factors_full(panel)

print("volume available assets:", list(vol.columns[vol.notna().any()]))

def factor_vol_z20():
    """volume z-score vs trailing 60d mean/std (log volume) - volume surge/starve"""
    lv = np.log(vol.replace(0, np.nan))
    mu = lv.rolling(60).mean(); sd = lv.rolling(60).std()
    return (lv - mu) / sd

def factor_vol_trend_20_60():
    """20d mean volume / 60d mean volume - volume expansion trend"""
    return vol.rolling(20).mean() / vol.rolling(60).mean()

def factor_amihud_20():
    """Amihud illiquidity: mean(|ret|/volume) over 20d, log - high = illiquid"""
    illiq = (ret.abs() / vol.replace(0, np.nan)).rolling(20).mean()
    return np.log(illiq)

def factor_vpt_20():
    """volume-price trend: 20d slope of OBV normalized - money flow direction"""
    obv = (np.sign(ret) * vol).cumsum()
    return obv.pct_change(20)

def factor_vol_ret_corr_20():
    """correlation of volume with return over 20d - volume-confirmed moves"""
    return ret.rolling(20).corr(vol)

cands = {
    'vol_z20': factor_vol_z20,
    'vol_trend_20_60': factor_vol_trend_20_60,
    'amihud_20_log': factor_amihud_20,
    'vpt_20': factor_vpt_20,
    'vol_ret_corr_20': factor_vol_ret_corr_20,
}

for name, fn in cands.items():
    try:
        fac = fn()
        res = eval_factor(fac, px, lib=lib)
        print_eval(name, res)
    except Exception as e:
        print(f"ERROR {name}: {e}")
    print()
