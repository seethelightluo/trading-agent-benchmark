"""miner_1 screening: novel factor families not yet in library (2026-07-30) - vectorized.
Universe: 15 tradable cross-asset instruments, warm-up window 2020-01-01..2026-07-15.
Candidates: trend efficiency, return autocorr, vol-price corr, RSI, DXY beta,
acceleration, sharpe momentum, skewness, range position.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_2_lib as lib

EPS = 1e-12
panel = lib.load_panel()
macro = lib.load_macro()

def _roll_acorr(r, win=20, minp=10):
    """Vectorized rolling lag-1 autocorrelation of daily returns."""
    y = r.shift(1)
    n = r.rolling(win, min_periods=minp).count()
    sx = r.rolling(win, min_periods=minp).sum()
    sy = y.rolling(win, min_periods=minp).sum()
    sxy = (r * y).rolling(win, min_periods=minp).sum()
    sxx = (r * r).rolling(win, min_periods=minp).sum()
    syy = (y * y).rolling(win, min_periods=minp).sum()
    num = n * sxy - sx * sy
    den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return num / (den + EPS)

def trend_eff_20d(panel, macro):
    """Kaufman efficiency ratio over 20d: net move / total path length."""
    def f(s):
        r = s.pct_change()
        path = r.abs().rolling(20, min_periods=10).sum()
        net = (s - s.shift(20)).abs()
        return net / (path + EPS)
    return lib.per_asset(f)(panel, macro)

def autocorr_20d(panel, macro):
    """Rolling 20d lag-1 autocorrelation of daily returns (per-asset)."""
    def f(s):
        return _roll_acorr(s.pct_change(), 20, 10)
    return lib.per_asset(f)(panel, macro)

def volprice_corr_20d(panel, macro):
    """Rolling 20d correlation between daily return and volume pct change."""
    out = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
        r = df["close"].pct_change()
        vc = df["volume"].pct_change()
        out[a] = r.rolling(20, min_periods=10).corr(vc)
    return pd.DataFrame(out, index=panel.index)

def rsi_14(panel, macro):
    """RSI-14 (per-asset)."""
    def f(s):
        r = s.pct_change()
        up = r.clip(lower=0).rolling(14, min_periods=7).mean()
        dn = (-r.clip(upper=0)).rolling(14, min_periods=7).mean()
        rs = up / (dn + EPS)
        return 100.0 - 100.0 / (1.0 + rs)
    return lib.per_asset(f)(panel, macro)

def dxy_beta_60d(panel, macro):
    """Rolling 60d beta of asset returns vs DXY returns (dollar sensitivity)."""
    dxy_r = macro["DXY"].pct_change()
    rets = panel.pct_change()
    beta = rets.rolling(60, min_periods=30).cov(dxy_r) / dxy_r.rolling(60, min_periods=30).var()
    return beta

def accel_mom_20x60(panel, macro):
    """Momentum acceleration: 20d mom (skip5) minus 60d mom (skip5)."""
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    m60 = panel.shift(5) / panel.shift(65) - 1.0
    return m20 - m60

def sharpe_mom_60d(panel, macro):
    """Risk-adjusted momentum: 60d mean daily return / 60d std."""
    r = panel.pct_change()
    return r.rolling(60, min_periods=30).mean() / (r.rolling(60, min_periods=30).std() + EPS)

def skew_20d(panel, macro):
    """Rolling 20d return skewness."""
    r = panel.pct_change()
    return r.rolling(20, min_periods=10).skew()

def range_pos_20d(panel, macro):
    """Close position within 20d high-low range [0,1]."""
    hi = panel.rolling(20, min_periods=10).max()
    lo = panel.rolling(20, min_periods=10).min()
    return (panel - lo) / (hi - lo + EPS)

CANDIDATES = [
    ("trend_eff_20d", trend_eff_20d),
    ("autocorr_20d", autocorr_20d),
    ("volprice_corr_20d", volprice_corr_20d),
    ("rsi_14", rsi_14),
    ("dxy_beta_60d", dxy_beta_60d),
    ("accel_mom_20x60", accel_mom_20x60),
    ("sharpe_mom_60d", sharpe_mom_60d),
    ("skew_20d", skew_20d),
    ("range_pos_20d", range_pos_20d),
]

import time
t0 = time.time()
for name, fn in CANDIDATES:
    try:
        res = lib.validate_factor(name, fn)
    except Exception as e:
        print(f"=== {name} FAILED: {e} ===")
print(f"total time {time.time()-t0:.1f}s")
