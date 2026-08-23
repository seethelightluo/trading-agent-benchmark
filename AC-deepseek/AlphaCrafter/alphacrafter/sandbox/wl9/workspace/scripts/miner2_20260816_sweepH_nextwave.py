"""miner_2 (2026-08-16): explore next wave of low-correlation candidates.

The library is crowded (momentum: mom_10/120; vol level: bb_width, vol_z,
rng_pos, skew; pullback age: days_since_high_60). This sweep targets NEW
economic dimensions orthogonal to the current library:
  - term-structure ratios (return/vol term slopes) -- regime timing
  - drawdown depth normalized (pullback MAGNITUDE, not age)
  - cross-asset relative momentum (asset vs SPX) -- beta-tilted momentum
  - volatility-of-return-asymmetry variants
  - EWMA-vol ratio with skip (already weak, try 10x60)

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on 15 assets;
persistence additionally requires max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes  # noqa: E402

closes = load_closes()


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()


def ret_term_ratio(close, short=5, long=60, skip=3):
    """(r_short - r_long)/vol_short: short-term return acceleration vs vol."""
    rs = close / close.shift(short + skip) - 1.0
    rl = close / close.shift(long) - 1.0
    vol = close.pct_change().rolling(short).std(ddof=0).replace(0, np.nan)
    return (rs - rl) / vol


def ret_vol_term_ratio(close, s_ret=20, l_ret=60, s_vol=10, l_vol=60):
    """Momentum slope vs volatility slope: (r20-r60)/(vol10-vol60)."""
    r20 = close / close.shift(s_ret) - 1.0
    r60 = close / close.shift(l_ret) - 1.0
    v10 = close.pct_change().rolling(s_vol).std(ddof=0)
    v60 = close.pct_change().rolling(l_vol).std(ddof=0)
    return (r20 - r60) / (v10 - v60).replace(0, np.nan)


def drawdown_depth_252(close, n=252):
    """Current depth below trailing n-day max, normalized by max."""
    rollmax = close.rolling(n, min_periods=n).max()
    return close / rollmax.replace(0, np.nan) - 1.0


def drawdown_depth_60(close, n=60):
    rollmax = close.rolling(n, min_periods=n).max()
    return close / rollmax.replace(0, np.nan) - 1.0


def rel_mom_spx(close, spx, n=20, skip=5):
    """Asset momentum minus SPX momentum: cross-sectional relative trend."""
    return (close / close.shift(n + skip) - 1.0) - (spx / spx.shift(n + skip) - 1.0)


def vol_asym_ratio(close, n=20, minp=4):
    """Ratio of realized vol of positive-return days to total vol (0.5=neutral)."""
    r = close.pct_change()
    pos = r.where(r > 0)
    vp = pos.rolling(n, min_periods=minp).std(ddof=0)
    vt = r.rolling(n, min_periods=minp).std(ddof=0)
    return vp / vt.replace(0, np.nan)


def hi_lo_skew_60(high, low, close, n=60, minp=20):
    """Skew of daily (high-low)/close ranges: range distribution tail."""
    rng = (high - low) / close.replace(0, np.nan)
    m = rng.rolling(n, min_periods=minp).mean()
    sd = rng.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    m3 = ((rng - m) ** 3).rolling(n, min_periods=minp).mean()
    return m3 / (sd ** 3)


def corr_btc_60(close, btc, n=60):
    """Rolling correlation of asset returns with BTC returns (crypto-beta)."""
    a = close.pct_change()
    b = btc.pct_change()
    return a.rolling(n).corr(b)


btc = closes["BTC"]
spx = closes["SPX"]

candidates = {
    "ret_term_ratio_5x60": {a: ret_term_ratio(closes[a], 5, 60, 3) for a in closes},
    "ret_vol_term_20x60": {a: ret_vol_term_ratio(closes[a]) for a in closes},
    "dd_depth_252": {a: drawdown_depth_252(closes[a], 252) for a in closes},
    "dd_depth_60": {a: drawdown_depth_60(closes[a], 60) for a in closes},
    "rel_mom_spx_20x5": {a: rel_mom_spx(closes[a], spx, 20, 5) for a in closes},
    "vol_asym_ratio_20d": {a: vol_asym_ratio(closes[a], 20) for a in closes},
    "hi_lo_skew_60d": {a: hi_lo_skew_60(ohlc[a]["high"].astype(float), ohlc[a]["low"].astype(float), closes[a], 60) for a in closes},
    "corr_btc_60": {a: corr_btc_60(closes[a], btc, 60) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()
