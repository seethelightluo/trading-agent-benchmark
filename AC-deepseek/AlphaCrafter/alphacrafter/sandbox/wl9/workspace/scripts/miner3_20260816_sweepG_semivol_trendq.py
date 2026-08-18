"""Exploration sweep G (miner_3, 2026-08-16): semi-vol asymmetry (fixed), trend
quality, candle close position, kurtosis, pullback age, EWMA vol term structure.

Bugs fixed from sweep D: downside_ratio/vol_pressure used .where(r>0) which leaves
mostly-NaN windows -> rolling std/mean with default min_periods=window returned NaN
everywhere. Here min_periods is set so semi-vol and up/down volume pressure are
actually computable.

Motivation: the library is crowded with momentum (mom_10/120, rng_pos, skew, bb_width
vol level, vol_z volume). This sweep targets SEPARATE economic dimensions:
- downside/upside semi-vol asymmetry (crash-risk asymmetry)
- trend efficiency / R2 (trend quality, not direction)
- daily candle close position (microstructure close strength)
- return kurtosis (tail risk)
- age of drawdown (pullback duration)
- EWMA volatility term-structure slope

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on the 15-asset universe;
persistence additionally requires max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes  # noqa: E402

VOL_ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"]
closes = load_closes()


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()


def semi_vol_ratio(close, n=20, minp=4):
    r = close.pct_change()
    neg = r.where(r < 0).rolling(n, min_periods=minp).std(ddof=0)
    pos = r.where(r > 0).rolling(n, min_periods=minp).std(ddof=0)
    return neg / pos.replace(0, np.nan)


def sortino_60(close, n=60, minp=5):
    r = close.pct_change()
    mom = close / close.shift(n) - 1.0
    downside = r.where(r < 0).rolling(n, min_periods=minp).std(ddof=0)
    return mom / downside.replace(0, np.nan)


def close_pos_20(high, low, close, n=20):
    pos = (close - low) / (high - low).replace(0, np.nan)
    return pos.rolling(n).mean()


def kurtosis_60(close, n=60, minp=30):
    r = close.pct_change()
    m = r.rolling(n, min_periods=minp).mean()
    sd = r.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    m4 = ((r - m) ** 4).rolling(n, min_periods=minp).mean()
    return m4 / (sd ** 4) - 3.0


def days_since_high_60(close, n=60):
    rollmax = close.rolling(n, min_periods=n).max()
    below = (close < rollmax).astype(float)
    # days since last at-high: use expanding count of consecutive below-high days
    out = pd.Series(np.nan, index=close.index)
    streak = 0
    vals = below.values
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            streak = 0
            out.iloc[i] = np.nan
        elif vals[i] == 0:
            streak = 0
            out.iloc[i] = 0.0
        else:
            streak += 1
            out.iloc[i] = float(streak)
    return out


def trend_r2_60(close, n=60):
    """R^2 of close vs linear time trend over n days, signed by trend direction."""
    r2s = close.rolling(n, min_periods=n).apply(
        lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2, raw=True
    )
    sign = np.sign(close - close.shift(n))
    return r2s * sign


def eff_ratio_60(close, n=60):
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n).sum()
    return num / den.replace(0, np.nan)


def ewm_vol_ratio(close, s=5, l=60):
    r = close.pct_change()
    vs = r.ewm(span=s, adjust=False).std()
    vl = r.ewm(span=l, adjust=False).std()
    return vs / vl.replace(0, np.nan)


def vol_pressure_20(close, volume, n=20, minp=4):
    r = close.pct_change()
    up = volume.where(r > 0).rolling(n, min_periods=minp).mean()
    dn = volume.where(r < 0).rolling(n, min_periods=minp).mean()
    return (up / dn.replace(0, np.nan)) - 1.0


vols = {a: ohlc[a]["volume"].astype(float) for a in VOL_ASSETS}

candidates = {
    "semi_vol_ratio_20d": {a: semi_vol_ratio(closes[a], 20) for a in closes},
    "sortino_60d": {a: sortino_60(closes[a], 60) for a in closes},
    "close_pos_20d": {a: close_pos_20(ohlc[a]["high"].astype(float), ohlc[a]["low"].astype(float), closes[a], 20) for a in closes},
    "kurtosis_60d": {a: kurtosis_60(closes[a], 60) for a in closes},
    "days_since_high_60": {a: days_since_high_60(closes[a], 60) for a in closes},
    "trend_r2_60_signed": {a: trend_r2_60(closes[a], 60) for a in closes},
    "eff_ratio_60": {a: eff_ratio_60(closes[a], 60) for a in closes},
    "ewm_vol_ratio_5x60": {a: ewm_vol_ratio(closes[a], 5, 60) for a in closes},
    "vol_pressure_20d_fix": {a: vol_pressure_20(closes[a], vols[a], 20) if a in vols else pd.Series(np.nan, index=closes[a].index) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()