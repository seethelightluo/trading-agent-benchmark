"""miner_2 (2026-08-19): sweep I - volume/liquidity + risk-adjusted return + trend-efficiency dimensions.

Library now: mom_10/120, bb_width, vol_z, rng_pos, skew, beta_VIX, days_since_high_60,
vix_beta_cond, (evicted: dside_ratio, vol_of_vol, bbz, beta_DXY, beta_USDJPY).
We target dimensions NOT yet covered:
  - volume z-score / volume trend (liquidity/attention)
  - Amihud illiquidity
  - return/vol (reward-to-risk, Sharpe-like) and return/downside-vol
  - Kaufman efficiency ratio (trend efficiency)
  - kurtosis of returns
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; persistence also needs max lib corr < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes

closes = load_closes()


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


ohlc = load_ohlc()

# ---- volume data quality ----
print("=== volume data quality ===")
for a in ASSETS:
    v = ohlc[a]["volume"].astype(float)
    nz = (v > 0).sum()
    print(f"  {a}: rows={len(v)} nonzero_vol={nz} frac={nz/len(v):.2f} last={v.dropna().iloc[-1] if nz else None}")


def vol_zscore(vol, n=20):
    """Volume z-score vs trailing n-day mean/std."""
    m = vol.rolling(n, min_periods=10).mean()
    s = vol.rolling(n, min_periods=10).std(ddof=0).replace(0, np.nan)
    return (vol - m) / s


def vol_ratio(vol, short=5, long=60):
    """Short/long volume ratio (volume trend)."""
    return vol.rolling(short).mean() / vol.rolling(long, min_periods=20).mean().replace(0, np.nan)


def amihud(close, vol, n=20):
    """|ret|/volume averaged over n days (illiquidity)."""
    r = close.pct_change().abs()
    illiq = (r / vol.replace(0, np.nan)).rolling(n, min_periods=10).mean()
    return -illiq  # negative = more liquid is higher


def ret_vol_ratio(close, n=60, skip=5):
    """Reward-to-risk: n-day return / n-day realized vol (Sharpe-like)."""
    r = close / close.shift(n + skip) - 1.0
    v = close.pct_change().rolling(n, min_periods=30).std(ddof=0).replace(0, np.nan)
    return r / v


def ret_downside_vol(close, n=60, skip=5):
    """Return / downside deviation (Sortino-like)."""
    r = close / close.shift(n + skip) - 1.0
    d = close.pct_change().clip(upper=0)
    dv = d.rolling(n, min_periods=30).std(ddof=0).replace(0, np.nan)
    return r / dv


def kaufman_eff(close, n=20):
    """Kaufman efficiency ratio: |close-close_n| / sum(|diffs|)."""
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n).sum().replace(0, np.nan)
    return num / den


def kurt_20d(close, n=20, minp=10):
    """Excess kurtosis of daily returns."""
    r = close.pct_change()
    m = r.rolling(n, min_periods=minp).mean()
    sd = r.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    m4 = ((r - m) ** 4).rolling(n, min_periods=minp).mean()
    return m4 / (sd ** 4) - 3.0


candidates = {
    "vol_z_20d_v2": {a: vol_zscore(ohlc[a]["volume"].astype(float), 20) for a in closes},
    "vol_ratio_5x60": {a: vol_ratio(ohlc[a]["volume"].astype(float), 5, 60) for a in closes},
    "amihud_illiq_20d": {a: amihud(closes[a], ohlc[a]["volume"].astype(float), 20) for a in closes},
    "ret_vol_60x5": {a: ret_vol_ratio(closes[a], 60, 5) for a in closes},
    "ret_downside_60x5": {a: ret_downside_vol(closes[a], 60, 5) for a in closes},
    "kaufman_eff_20d": {a: kaufman_eff(closes[a], 20) for a in closes},
    "kurt_20d": {a: kurt_20d(closes[a], 20) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()
